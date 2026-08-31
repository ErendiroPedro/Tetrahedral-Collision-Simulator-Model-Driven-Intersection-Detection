"""
Tetrahedral Collision Simulator

Detects collisions between two tetrahedral meshes with a two-stage pipeline:
1. Broad phase: R-tree over per-tetrahedron AABBs (``trimesh.util.bounds_tree``).
2. Narrow phase: batched TorchScript inference over a configurable transform chain.
3. Visualization: polyscope volume meshes with colliding cells highlighted.

Transforms are registered in ``TRANSFORMS`` and selected with ``--transform``; each
bundled model declares its expected chain in ``MODEL_TRANSFORMS``.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
import trimesh
import trimesh.util

logger = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class _CompactFormatter(logging.Formatter):
    """Drops the level prefix on INFO so per-step progress lines stay scannable."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        return message if record.levelno == logging.INFO else f"{record.levelname}: {message}"


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
        if level <= logging.DEBUG
        else _CompactFormatter()
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


@contextlib.contextmanager
def quiet_native_stdout(enabled: bool = True) -> Iterator[None]:
    """Silence writes to fd 1; polyscope prints its backend banner there, bypassing Python."""
    if not enabled:
        yield
        return
    saved, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)


# --------------------------------------------------------------------------- #
# Model input transforms
# --------------------------------------------------------------------------- #

Transform = Callable[[torch.Tensor], torch.Tensor]


def _split_pair(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a flat (batch, 24) pair encoding into two (batch, 4, 3) tetrahedra."""
    pair = data.to(torch.float64).reshape(-1, 8, 3)
    return pair[:, :4], pair[:, 4:]


def principal_axis(data: torch.Tensor) -> torch.Tensor:
    """Re-express both tetrahedra in the principal-axis frame of the first. -> (batch, 24)"""
    tetra1, tetra2 = _split_pair(data)
    centroid = tetra1.mean(dim=1, keepdim=True)
    tetra1, tetra2 = tetra1 - centroid, tetra2 - centroid
    _, axes = torch.linalg.eigh(torch.bmm(tetra1.transpose(1, 2), tetra1))
    return torch.cat([torch.bmm(tetra1, axes), torch.bmm(tetra2, axes)], dim=1).flatten(1)


def unit_frame(data: torch.Tensor) -> torch.Tensor:
    """Map the first tetrahedron to the unit simplex, keeping only the second. -> (batch, 12)"""
    tetra1, tetra2 = _split_pair(data)
    origin = tetra1[:, :1]
    edges = tetra1[:, 1:] - origin
    relative = tetra2 - origin
    try:
        relative = torch.linalg.solve(edges.transpose(1, 2), relative.transpose(1, 2)).transpose(1, 2)
    except torch.linalg.LinAlgError:  # degenerate tetrahedron: keep the translated coordinates
        pass
    return relative.flatten(1)


TRANSFORMS: Dict[str, Transform] = {
    "pca": principal_axis,
    "unit": unit_frame,
}

# Chain each bundled model was trained with, keyed by model file stem.
MODEL_TRANSFORMS: Dict[str, Tuple[str, ...]] = {
    "EuroGraphicsPaper_PCA": ("pca",),
    "EuroGraphicsDemo_PCA_Unit": ("pca", "unit"),
}
DEFAULT_TRANSFORMS: Tuple[str, ...] = ("pca",)


def transforms_for_model(model_path: str) -> Tuple[str, ...]:
    return MODEL_TRANSFORMS.get(Path(model_path).stem, DEFAULT_TRANSFORMS)


def build_pipeline(names: Sequence[str]) -> Transform:
    unknown = sorted(set(names) - set(TRANSFORMS))
    if unknown:
        raise ValueError(f"Unknown transform(s) {unknown}; available: {sorted(TRANSFORMS)}")
    steps = [TRANSFORMS[name] for name in names]

    def pipeline(data: torch.Tensor) -> torch.Tensor:
        for step in steps:
            data = step(data)
        return data

    return pipeline


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    num_steps: int = 50
    batch_size: int = 2048
    initial_offset: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.0, 0.0]))
    rotate_static: bool = False
    rotation_speed: float = np.radians(2.0)  # radians per step
    rotation_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    collision_color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass
class TetrahedralMesh:
    """A tetrahedralized mesh and the per-tetrahedron AABBs used by the broad phase."""
    name: str
    color: List[float]
    vertices: np.ndarray
    tetrahedra: np.ndarray
    bounds: np.ndarray = field(init=False)

    def __post_init__(self):
        self._refresh_bounds()

    @classmethod
    def from_file(cls, path: str, name: str, color: Sequence[float],
                  offset: Optional[np.ndarray] = None) -> "TetrahedralMesh":
        import tetgen

        surface = trimesh.load(path, force="mesh")
        if offset is not None:
            surface.apply_translation(offset)
        tet = tetgen.TetGen(surface.vertices, surface.faces)
        # Newer tetgen builds return extra trailing arrays; only nodes and elements are needed.
        vertices, tetrahedra = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, quality=True)[:2]
        logger.info("%-6s %8s tetrahedra from %s vertices", name, f"{len(tetrahedra):,}", f"{len(vertices):,}")
        return cls(name, list(color), np.asarray(vertices, dtype=np.float64), np.asarray(tetrahedra))

    def _refresh_bounds(self):
        corners = self.vertices[self.tetrahedra]
        self.bounds = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)

    def set_vertices(self, vertices: np.ndarray):
        self.vertices = vertices
        self._refresh_bounds()

    def translate(self, offset: np.ndarray):
        self.set_vertices(self.vertices + offset)

    def apply_transform(self, matrix: np.ndarray):
        self.set_vertices(trimesh.transform_points(self.vertices, matrix))

    def corners_of(self, indices: np.ndarray) -> np.ndarray:
        """Flatten the 4 vertices of the selected tetrahedra into (n, 12) rows."""
        return self.vertices[self.tetrahedra[indices]].reshape(-1, 12)


# --------------------------------------------------------------------------- #
# Narrow-phase model
# --------------------------------------------------------------------------- #

# One non-degenerate pair of unit tetrahedra, used to validate model/transform pairing.
_PROBE_PAIR = torch.tensor(
    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2], dtype=torch.float64
)


class CollisionDetector:
    """Classifies tetrahedron pairs with a TorchScript model fed by a transform chain."""

    def __init__(self, model_path: str, transforms: Sequence[str], device: str = "cpu"):
        self.device = device
        self.transforms = tuple(transforms)
        self.pipeline = build_pipeline(self.transforms)
        self.model = torch.jit.load(model_path, map_location=device).double().eval()
        try:
            self.classify(_PROBE_PAIR)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{model_path}' rejects the transform chain {self.transforms}. "
                f"Known pairings: {MODEL_TRANSFORMS}"
            ) from exc
        logger.info("Loaded %s on %s with transforms %s", model_path, device, self.transforms)

    def classify(self, pairs: torch.Tensor) -> np.ndarray:
        """True where the model predicts the tetrahedron pair intersects."""
        with torch.no_grad():
            logits = self.model(self.pipeline(pairs.to(self.device)))
        # The bundled models emit two logits per pair; class 0 means intersecting.
        return (logits.argmax(dim=1) == 0).cpu().numpy()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


class VisualizationManager:
    """Manages Polyscope volume mesh visualization."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.headless = False

    def initialize(self, force_headless: bool = False):
        quiet = not logger.isEnabledFor(logging.DEBUG)
        if not force_headless:
            try:
                with quiet_native_stdout(quiet):
                    ps.init()
                self.headless = ps.is_headless()
                return
            except RuntimeError as e:
                logger.warning("Interactive Polyscope backend unavailable (%s); falling back to headless EGL.", e)
        ps.set_allow_headless_backends(True)
        with quiet_native_stdout(quiet):
            ps.init("openGL3_egl")
        self.headless = ps.is_headless()
        logger.info("Polyscope running headless; frames must be captured with screenshots.")

    def register(self, mesh: TetrahedralMesh):
        ps.register_volume_mesh(mesh.name, mesh.vertices, mesh.tetrahedra).set_color(mesh.color)

    def sync_positions(self, mesh: TetrahedralMesh):
        ps.get_volume_mesh(mesh.name).update_vertex_positions(mesh.vertices)

    def shade_collisions(self, mesh: TetrahedralMesh, colliding: np.ndarray):
        colors = np.tile(mesh.color, (len(mesh.tetrahedra), 1))
        colors[colliding] = self.config.collision_color
        ps.get_volume_mesh(mesh.name).add_color_quantity(
            "collisions", colors, defined_on="cells", enabled=True
        )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

_NO_PAIRS = np.empty((0, 2), dtype=np.int64)


@dataclass
class StepStats:
    """Per-step pipeline metrics backing the progress line and the run summary."""
    step: int
    candidates: int
    collisions: int
    broad_s: float
    narrow_s: float

    @property
    def detection_s(self) -> float:
        return self.broad_s + self.narrow_s


class TetrahedralCollisionSimulator:
    """Drives the moving mesh toward the static mesh and reports colliding tetrahedra."""

    def __init__(self, config: SimulationConfig, detector: CollisionDetector,
                 static_mesh: TetrahedralMesh, moving_mesh: TetrahedralMesh):
        self.config = config
        self.detector = detector
        self.static_mesh = static_mesh
        self.moving_mesh = moving_mesh
        self.visualization_manager = VisualizationManager(config)
        self.step_size = -config.initial_offset / config.num_steps
        self.home_vertices = [mesh.vertices.copy() for mesh in self.meshes]
        self.current_step = 0
        self.candidate_pairs = _NO_PAIRS
        self.collision_pairs = _NO_PAIRS
        self.is_playing = False
        self.history: List[StepStats] = []
        self.run_started: Optional[float] = None

    @property
    def meshes(self) -> Tuple[TetrahedralMesh, TetrahedralMesh]:
        return self.static_mesh, self.moving_mesh

    def broad_phase(self) -> np.ndarray:
        """Return (n, 2) AABB-overlapping tetrahedron indices into the static/moving meshes."""
        tree = trimesh.util.bounds_tree(self.static_mesh.bounds)
        pairs = {
            (int(index_static), index_moving)
            for index_moving, box in enumerate(self.moving_mesh.bounds)
            for index_static in tree.intersection(box)
        }
        return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)

    def narrow_phase(self, candidates: np.ndarray) -> np.ndarray:
        """Return the subset of `candidates` the model classifies as intersecting."""
        if len(candidates) == 0:
            return candidates
        pair_data = torch.from_numpy(np.concatenate(
            [mesh.corners_of(candidates[:, column]) for column, mesh in enumerate(self.meshes)],
            axis=1,
        ))
        hits = np.concatenate([
            self.detector.classify(pair_data[offset:offset + self.config.batch_size])
            for offset in range(0, len(candidates), self.config.batch_size)
        ])
        return candidates[hits]

    def detect_collisions(self) -> Tuple[float, float]:
        start = time.perf_counter()
        self.candidate_pairs = self.broad_phase()
        broad_s = time.perf_counter() - start

        start = time.perf_counter()
        self.collision_pairs = self.narrow_phase(self.candidate_pairs)
        return broad_s, time.perf_counter() - start

    def refresh_visualization(self):
        for column, mesh in enumerate(self.meshes):
            self.visualization_manager.sync_positions(mesh)
            self.visualization_manager.shade_collisions(mesh, self.collision_pairs[:, column])

    def step_forward(self):
        if self.current_step >= self.config.num_steps:
            self.is_playing = False
            return
        if self.run_started is None:
            self.run_started = time.perf_counter()

        self.moving_mesh.translate(self.step_size)
        if self.config.rotate_static:
            rotation = trimesh.transformations.rotation_matrix(
                self.config.rotation_speed,
                self.config.rotation_axis,
                point=self.static_mesh.vertices.mean(axis=0),
            )
            self.static_mesh.apply_transform(rotation)

        broad_s, narrow_s = self.detect_collisions()
        self.refresh_visualization()
        self.current_step += 1
        self._log_step(broad_s, narrow_s)

        if self.current_step >= self.config.num_steps:
            self.log_summary()

    def _log_step(self, broad_s: float, narrow_s: float):
        stats = StepStats(self.current_step, len(self.candidate_pairs),
                          len(self.collision_pairs), broad_s, narrow_s)
        self.history.append(stats)
        total_candidates = sum(s.candidates for s in self.history)
        detection_s = sum(s.detection_s for s in self.history)
        aggregate = (f"{detection_s / total_candidates * 1e6:6.2f}us/cand"
                     if total_candidates else "-".center(13))
        logger.info(
            "step %3d/%-3d  cand %11s  hit %11s  broad %7.3fs  narrow %7.3fs  elapsed %7.1fs  agg %s",
            stats.step, self.config.num_steps, f"{stats.candidates:,}", f"{stats.collisions:,}",
            stats.broad_s, stats.narrow_s, time.perf_counter() - self.run_started, aggregate,
        )

    def log_summary(self):
        if not self.history:
            return
        total_candidates = sum(s.candidates for s in self.history)
        total_collisions = sum(s.collisions for s in self.history)
        detection_s = sum(s.detection_s for s in self.history)
        busiest = max(self.history, key=lambda s: s.candidates)
        slowest = max(self.history, key=lambda s: s.detection_s)

        logger.info("")
        logger.info("run summary   %d steps  %s candidates  %s collisions",
                    len(self.history), f"{total_candidates:,}", f"{total_collisions:,}")
        for label, attr in (("broad", "broad_s"), ("narrow", "narrow_s")):
            get = attrgetter(attr)
            times = [get(s) for s in self.history]
            peak = max(self.history, key=get)
            logger.info("  %-6s      %8.2fs total  %7.3fs mean  %7.3fs peak (step %d)",
                        label, sum(times), sum(times) / len(times), get(peak), peak.step)
        broad_share = 100.0 * sum(s.broad_s for s in self.history) / detection_s if detection_s else 0.0
        logger.info("  detection   %8.2fs total  %s  broad %.0f%% / narrow %.0f%%",
                    detection_s,
                    f"{detection_s / total_candidates * 1e6:.2f}us/cand aggregate"
                    if total_candidates else "no candidates",
                    broad_share, 100.0 - broad_share)
        logger.info("  busiest     step %d (%s candidates)   slowest  step %d (%.2fs)",
                    busiest.step, f"{busiest.candidates:,}", slowest.step, slowest.detection_s)

    def reset_simulation(self):
        self.is_playing = False
        self.current_step = 0
        self.candidate_pairs = _NO_PAIRS
        self.collision_pairs = _NO_PAIRS
        self.history.clear()
        self.run_started = None
        for mesh, home in zip(self.meshes, self.home_vertices):
            mesh.set_vertices(home.copy())
        self.refresh_visualization()
        logger.info("Simulation reset.")

    def initialize_visualization(self, force_headless: bool = False):
        self.visualization_manager.initialize(force_headless)
        for mesh in self.meshes:
            self.visualization_manager.register(mesh)

    def create_ui_callback(self):
        def ui_callback():
            psim.Text(f"Step: {self.current_step}/{self.config.num_steps}")
            psim.Text(f"Transforms: {' -> '.join(self.detector.transforms)}")

            if self.is_playing:
                if psim.Button("Stop"):
                    self.is_playing = False
            elif psim.Button("Play") and self.current_step < self.config.num_steps:
                self.is_playing = True

            psim.SameLine()
            if psim.Button("Reset"):
                self.reset_simulation()

            psim.Separator()
            psim.Text(f"Candidate Pairs: {len(self.candidate_pairs)}")
            psim.Text(f"Colliding Pairs: {len(self.collision_pairs)}")

            if self.is_playing:
                self.step_forward()

        return ui_callback


STATIC_COLOR = [0.3, 0.3, 1.0]
MOVING_COLOR = [0.3, 1.0, 0.3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the tetrahedral collision simulator")
    parser.add_argument("--model", "-M", type=str, default="model/EuroGraphicsDemo_PCA_Unit.pt",
                        help="Path to the TorchScript collision model")
    parser.add_argument("--transform", "-T", nargs="+", choices=sorted(TRANSFORMS), default=None,
                        metavar="NAME",
                        help=f"Transform chain applied before inference, in order "
                             f"(choices: {sorted(TRANSFORMS)}); defaults to the chain "
                             f"registered for the selected model")
    parser.add_argument("--mesh-a", type=str, default="data/armadillo.obj",
                        help="Path to the first (static) mesh OBJ file")
    parser.add_argument("--mesh-b", type=str, default=None,
                        help="Path to the second (moving) mesh OBJ file; defaults to the same file as --mesh-a")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of simulation steps")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Tetrahedron pairs per inference batch")
    parser.add_argument("--offset-x", type=float, default=2.5,
                        help="Initial x offset for the moving mesh")
    parser.add_argument("--rotate-static", action="store_true",
                        help="Rotate the static mesh while the second mesh moves")
    parser.add_argument("--rotation-speed", type=float, default=np.radians(10.0),
                        help="Static mesh rotation speed in radians per step")
    parser.add_argument("--rotation-axis", type=float, nargs=3, default=[0.0, 1.0, 0.0],
                        help="Rotation axis for the static mesh")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution even if CUDA is available")
    parser.add_argument("--headless", action="store_true",
                        help="Skip the interactive window and render every step to PNG screenshots")
    parser.add_argument("--screenshot-dir", type=str, default="screenshots",
                        help="Directory for headless screenshots")
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="INFO",
                        help="Logging verbosity; DEBUG adds timestamps and reveals polyscope's banner")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level)

    config = SimulationConfig(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        initial_offset=np.array([args.offset_x, 0.0, 0.0]),
        rotate_static=args.rotate_static,
        rotation_speed=args.rotation_speed,
        rotation_axis=np.array(args.rotation_axis),
    )

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    transforms = tuple(args.transform) if args.transform else transforms_for_model(args.model)
    detector = CollisionDetector(args.model, transforms, device=device)

    static_mesh = TetrahedralMesh.from_file(args.mesh_a, "Mesh A", STATIC_COLOR)
    moving_mesh = TetrahedralMesh.from_file(args.mesh_b or args.mesh_a, "Mesh B", MOVING_COLOR,
                                            offset=config.initial_offset)

    simulator = TetrahedralCollisionSimulator(config, detector, static_mesh, moving_mesh)
    simulator.initialize_visualization(force_headless=args.headless)

    if simulator.visualization_manager.headless:
        out_dir = Path(args.screenshot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ps.screenshot(str(out_dir / "step_000.png"))
        for _ in range(config.num_steps):
            simulator.step_forward()
            ps.screenshot(str(out_dir / f"step_{simulator.current_step:03d}.png"))
        logger.info("Wrote %d frames to %s", config.num_steps + 1, out_dir.resolve())
        return

    ps.set_user_callback(simulator.create_ui_callback())
    ps.show()


if __name__ == "__main__":
    main()
