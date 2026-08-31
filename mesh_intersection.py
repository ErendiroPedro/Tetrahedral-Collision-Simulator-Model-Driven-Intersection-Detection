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
import ctypes
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
from rtree import core as rtree_core

logger = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class _CompactFormatter(logging.Formatter):
    """Drops the level prefix on INFO so per-step progress lines stay scannable."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record``, prefixing only messages above the INFO level."""
        message = record.getMessage()
        return (
            message
            if record.levelno == logging.INFO
            else f"{record.levelname}: {message}"
        )


def configure_logging(level_name: str) -> None:
    """Configure console logging for a named level and replace inherited handlers."""
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
    """Optionally redirect native writes to stdout while the context is active.

    Polyscope prints its backend banner directly to file descriptor 1, bypassing Python's
    logging streams, so temporarily replacing that descriptor is the only reliable mute.
    """
    if not enabled:
        yield
        return
    saved_stdout_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 1)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.close(null_fd)
        os.close(saved_stdout_fd)


# --------------------------------------------------------------------------- #
# Model input transforms
# --------------------------------------------------------------------------- #

Transform = Callable[[torch.Tensor], torch.Tensor]


def _split_pair(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the static and moving tetrahedra encoded by flat ``(batch, 24)`` rows."""
    tetrahedron_pairs = data.reshape(-1, 8, 3)
    static_tetrahedra = tetrahedron_pairs[:, :4]
    moving_tetrahedra = tetrahedron_pairs[:, 4:]
    return static_tetrahedra, moving_tetrahedra


def principal_axis(data: torch.Tensor) -> torch.Tensor:
    """Express each pair in the static tetrahedron's principal-axis coordinate frame.

    ``data`` contains flat ``(batch, 24)`` tetrahedron pairs. The returned tensor has
    the same shape; translating and rotating both tetrahedra together leaves it unchanged.
    """
    static_tetrahedra, moving_tetrahedra = _split_pair(data)
    static_centroids = static_tetrahedra.mean(dim=1, keepdim=True)
    centered_static_vertices = static_tetrahedra - static_centroids
    centered_moving_vertices = moving_tetrahedra - static_centroids

    covariance_matrices = torch.bmm(
        centered_static_vertices.transpose(1, 2), centered_static_vertices
    )
    _, principal_axes = torch.linalg.eigh(covariance_matrices)
    aligned_static_vertices = torch.bmm(centered_static_vertices, principal_axes)
    aligned_moving_vertices = torch.bmm(centered_moving_vertices, principal_axes)

    aligned_pairs = torch.cat([aligned_static_vertices, aligned_moving_vertices], dim=1)
    return aligned_pairs.flatten(1)


def unit_frame(data: torch.Tensor) -> torch.Tensor:
    """Map moving vertices into the static tetrahedron's unit-simplex frame.

    ``data`` contains flat ``(batch, 24)`` tetrahedron pairs. The static tetrahedron
    defines the origin and basis; the returned ``(batch, 12)`` tensor contains only
    the four transformed moving vertices expected by the PCA/unit model.
    """
    static_tetrahedra, moving_tetrahedra = _split_pair(data)
    static_origins = static_tetrahedra[:, :1]
    static_edges = static_tetrahedra[:, 1:] - static_origins
    relative_positions = moving_tetrahedra - static_origins

    edge_matrices = static_edges.transpose(1, 2)
    relative_position_columns = relative_positions.transpose(1, 2)
    try:
        unit_frame_columns = torch.linalg.solve(
            edge_matrices, relative_position_columns
        )
        relative_positions = unit_frame_columns.transpose(1, 2)
    except torch.linalg.LinAlgError:  # degenerate tetrahedron: keep the translated coordinates
        pass
    return relative_positions.flatten(1)


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
    """Return the registered transform chain for ``model_path`` or the default chain."""
    return MODEL_TRANSFORMS.get(Path(model_path).stem, DEFAULT_TRANSFORMS)


def build_pipeline(names: Sequence[str]) -> Transform:
    """Build a callable that applies registered transform ``names`` from left to right."""
    unknown = sorted(set(names) - set(TRANSFORMS))
    if unknown:
        raise ValueError(
            f"Unknown transform(s) {unknown}; available: {sorted(TRANSFORMS)}"
        )
    steps = [TRANSFORMS[name] for name in names]

    def pipeline(data: torch.Tensor) -> torch.Tensor:
        """Return ``data`` after applying each configured transform in sequence."""
        for step in steps:
            data = step(data)
        return data

    return pipeline


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class SimulationConfig:
    """Parameters that control mesh motion, inference batching, and collision color."""

    num_steps: int = 50
    batch_size: int = 2048
    initial_offset: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 0.0, 0.0])
    )
    rotate_static: bool = False
    rotation_speed: float = np.radians(2.0)  # radians per step
    rotation_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    collision_color: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved command-line settings needed to construct and run the simulator."""

    model_path: str
    transform_names: Tuple[str, ...]
    static_mesh_path: str
    moving_mesh_path: str
    device: str
    dtype: torch.dtype
    simulation: SimulationConfig
    headless: bool
    screenshot_dir: Path
    log_level: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RuntimeConfig":
        """Resolve device-dependent defaults and convert CLI values into runtime types."""
        device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

        if args.float64:
            dtype = torch.float64
        elif args.float32 or device == "cuda":
            dtype = torch.float32
        else:
            dtype = torch.float64

        batch_size = args.batch_size or (65536 if device == "cuda" else 2048)
        simulation = SimulationConfig(
            num_steps=args.num_steps,
            batch_size=batch_size,
            initial_offset=np.array([args.offset_x, 0.0, 0.0]),
            rotate_static=args.rotate_static,
            rotation_speed=args.rotation_speed,
            rotation_axis=np.array(args.rotation_axis),
        )
        transform_names = (
            tuple(args.transform)
            if args.transform
            else transforms_for_model(args.model)
        )

        return cls(
            model_path=args.model,
            transform_names=transform_names,
            static_mesh_path=args.mesh_a,
            moving_mesh_path=args.mesh_b or args.mesh_a,
            device=device,
            dtype=dtype,
            simulation=simulation,
            headless=args.headless,
            screenshot_dir=Path(args.screenshot_dir),
            log_level=args.log_level,
        )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass
class TetrahedralMesh:
    """Store tetrahedral geometry and the per-cell AABBs used by the broad phase."""

    name: str
    color: List[float]
    vertices: np.ndarray
    tetrahedra: np.ndarray
    bounds: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Derive broad-phase bounds after dataclass initialization."""
        self._refresh_bounds()

    @classmethod
    def from_file(
        cls,
        path: str,
        name: str,
        color: Sequence[float],
        offset: Optional[np.ndarray] = None,
    ) -> "TetrahedralMesh":
        """Tetrahedralize a surface mesh and return it with optional translation applied."""
        import tetgen

        surface = trimesh.load(path, force="mesh")
        if offset is not None:
            surface.apply_translation(offset)
        tet = tetgen.TetGen(surface.vertices, surface.faces)
        # Newer tetgen builds return extra trailing arrays; only nodes and elements are needed.
        vertices, tetrahedra = tet.tetrahedralize(
            order=1, mindihedral=20, minratio=1.5, quality=True
        )[:2]
        logger.info(
            "%-6s %8s tetrahedra from %s vertices",
            name,
            f"{len(tetrahedra):,}",
            f"{len(vertices):,}",
        )
        return cls(
            name,
            list(color),
            np.asarray(vertices, dtype=np.float64),
            np.asarray(tetrahedra),
        )

    def _refresh_bounds(self) -> None:
        """Recompute one axis-aligned bounding box per tetrahedron from current vertices."""
        corners = self.vertices[self.tetrahedra]
        self.bounds = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)

    def set_vertices(self, vertices: np.ndarray) -> None:
        """Replace all vertex positions and rebuild the derived broad-phase bounds."""
        self.vertices = vertices
        self._refresh_bounds()

    def translate(self, offset: np.ndarray) -> None:
        """Translate vertices and cached AABBs in place by the three-vector ``offset``."""
        self.vertices += offset
        self.bounds[:, :3] += offset
        self.bounds[:, 3:] += offset

    def apply_transform(self, matrix: np.ndarray) -> None:
        """Apply a homogeneous transform to all vertices and rebuild their AABBs."""
        self.set_vertices(trimesh.transform_points(self.vertices, matrix))

    def corners_of(self, indices: np.ndarray) -> np.ndarray:
        """Flatten the 4 vertices of the selected tetrahedra into (n, 12) rows."""
        return self.vertices[self.tetrahedra[indices]].reshape(-1, 12)


# --------------------------------------------------------------------------- #
# Narrow-phase model
# --------------------------------------------------------------------------- #

# One non-degenerate pair of unit tetrahedra, used to validate model/transform pairing.
_PROBE_PAIR = torch.tensor(
    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * 2],
    dtype=torch.float64,
)


@dataclass
class _PcaUnitStaticFrames:
    """Static tetrahedron frames cached on the inference device for PCA/unit transforms."""

    static_centroids: torch.Tensor
    principal_axes: torch.Tensor
    aligned_origins: torch.Tensor
    edge_matrices: torch.Tensor


class CollisionDetector:
    """Classify tetrahedron pairs with a TorchScript model and registered transforms.

    The detector owns the model, its inference device and dtype, and optional static-mesh
    frame data used to avoid recomputing PCA/unit transforms for every candidate pair.
    """

    def __init__(
        self,
        model_path: str,
        transforms: Sequence[str],
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        """Load ``model_path`` on ``device`` and verify its configured input transforms."""
        self.device = device
        self.is_cuda = torch.device(device).type == "cuda"
        self.dtype = dtype
        self.transforms = tuple(transforms)
        self.pipeline = build_pipeline(self.transforms)
        self.model = (
            torch.jit.load(model_path, map_location=device).to(dtype=dtype).eval()
        )
        self.pca_unit_frames: Optional[_PcaUnitStaticFrames] = None
        try:
            self.classify(_PROBE_PAIR)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{model_path}' rejects the transform chain {self.transforms}. "
                f"Known pairings: {MODEL_TRANSFORMS}"
            ) from exc
        logger.info(
            "Loaded %s on %s as %s with transforms %s",
            model_path,
            device,
            str(dtype).removeprefix("torch."),
            self.transforms,
        )

    def classify(self, pairs: torch.Tensor) -> np.ndarray:
        """Return a NumPy mask that is true for each predicted intersection in ``pairs``."""
        return self.classify_tensor(pairs).cpu().numpy()

    def classify_tensor(self, pairs: torch.Tensor) -> torch.Tensor:
        """Transform ``pairs`` and return their collision mask on the detector device."""
        with torch.inference_mode():
            model_inputs = self.pipeline(pairs.to(device=self.device, dtype=self.dtype))
            logits = self.model(model_inputs)
        # The bundled models emit two logits per pair; class 0 means intersecting.
        return logits.argmax(dim=1) == 0

    def prepare_static(self, tetrahedra: np.ndarray) -> None:
        """Cache PCA/unit frames for static ``(tetrahedra, 4, 3)`` vertices.

        Other transform chains cannot use this specialization, so calling this method
        clears any previous cache and returns without allocating device tensors.
        """
        self.pca_unit_frames = None
        if self.transforms != ("pca", "unit"):
            return

        with torch.inference_mode():
            static_tetrahedra = torch.as_tensor(
                tetrahedra, dtype=self.dtype, device=self.device
            )
            static_centroids = static_tetrahedra.mean(dim=1, keepdim=True)
            centered_static_vertices = static_tetrahedra - static_centroids
            covariance_matrices = torch.bmm(
                centered_static_vertices.transpose(1, 2), centered_static_vertices
            )
            _, principal_axes = torch.linalg.eigh(covariance_matrices)
            aligned_static_vertices = torch.bmm(
                centered_static_vertices, principal_axes
            )
            aligned_origins = aligned_static_vertices[:, :1]
            aligned_edges = aligned_static_vertices[:, 1:] - aligned_origins
            edge_matrices = aligned_edges.transpose(1, 2)

        self.pca_unit_frames = _PcaUnitStaticFrames(
            static_centroids, principal_axes, aligned_origins, edge_matrices
        )

    def classify_pca_unit(
        self, static_indices: np.ndarray, moving_tetrahedra: np.ndarray
    ) -> np.ndarray:
        """Return a NumPy collision mask using cached frames for static candidates."""
        return (
            self.classify_pca_unit_tensor(static_indices, moving_tetrahedra)
            .cpu()
            .numpy()
        )

    def classify_pca_unit_tensor(
        self, static_indices: torch.Tensor, moving_tetrahedra: torch.Tensor
    ) -> torch.Tensor:
        """Classify a candidate batch with cached static PCA/unit frame tensors.

        ``static_indices`` selects one cached frame per ``(batch, 4, 3)`` moving
        tetrahedron. The returned Boolean tensor remains on the detector device so CUDA
        callers can concatenate all batches before performing one host transfer.
        """
        if self.pca_unit_frames is None:
            raise RuntimeError("Static PCA/unit frames have not been prepared")

        with torch.inference_mode():
            static_batch_indices = torch.as_tensor(
                static_indices, dtype=torch.long, device=self.device
            )
            moving_batch_vertices = torch.as_tensor(
                moving_tetrahedra, dtype=self.dtype, device=self.device
            )

            static_centroids = self.pca_unit_frames.static_centroids[
                static_batch_indices
            ]
            principal_axes = self.pca_unit_frames.principal_axes[static_batch_indices]
            aligned_origins = self.pca_unit_frames.aligned_origins[static_batch_indices]
            edge_matrices = self.pca_unit_frames.edge_matrices[static_batch_indices]

            centered_moving_vertices = moving_batch_vertices - static_centroids
            aligned_moving_vertices = torch.bmm(
                centered_moving_vertices, principal_axes
            )
            relative_positions = aligned_moving_vertices - aligned_origins
            relative_position_columns = relative_positions.transpose(1, 2)
            try:
                unit_frame_columns = torch.linalg.solve(
                    edge_matrices, relative_position_columns
                )
                relative_positions = unit_frame_columns.transpose(1, 2)
            except torch.linalg.LinAlgError:
                pass

            model_inputs = relative_positions.flatten(1)
            logits = self.model(model_inputs)
        return logits.argmax(dim=1) == 0


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #


class VisualizationManager:
    """Register simulation meshes with Polyscope and update their rendered quantities."""

    def __init__(self, config: SimulationConfig):
        """Retain visual settings from ``config`` until a Polyscope backend is initialized."""
        self.config = config
        self.headless = False

    def initialize(self, force_headless: bool = False) -> None:
        """Initialize an interactive backend, falling back to headless EGL when needed."""
        quiet = not logger.isEnabledFor(logging.DEBUG)
        if not force_headless:
            try:
                with quiet_native_stdout(quiet):
                    ps.init()
                self.headless = ps.is_headless()
                return
            except RuntimeError as e:
                logger.warning(
                    "Interactive Polyscope backend unavailable (%s); falling back to headless EGL.",
                    e,
                )
        ps.set_allow_headless_backends(True)
        with quiet_native_stdout(quiet):
            ps.init("openGL3_egl")
        self.headless = ps.is_headless()
        logger.info(
            "Polyscope running headless; frames must be captured with screenshots."
        )

    def register(self, mesh: TetrahedralMesh) -> None:
        """Register ``mesh`` as a colored Polyscope volume mesh."""
        ps.register_volume_mesh(mesh.name, mesh.vertices, mesh.tetrahedra).set_color(
            mesh.color
        )

    def sync_positions(self, mesh: TetrahedralMesh) -> None:
        """Copy the current vertices of ``mesh`` into its registered render object."""
        ps.get_volume_mesh(mesh.name).update_vertex_positions(mesh.vertices)

    def shade_collisions(self, mesh: TetrahedralMesh, colliding: np.ndarray) -> None:
        """Color cells indexed by ``colliding`` with the configured collision color."""
        colors = np.tile(mesh.color, (len(mesh.tetrahedra), 1))
        colors[colliding] = self.config.collision_color
        ps.get_volume_mesh(mesh.name).add_color_quantity(
            "collisions", colors, defined_on="cells", enabled=True
        )


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #

_NO_PAIRS = np.empty((0, 2), dtype=np.int64)


def _bulk_tree_intersections(tree, bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return all Rtree intersections for ``(n, 6)`` AABBs in one vector query.

    The first returned array stores matching tree IDs, grouped by query. The second
    stores the number of IDs in each group. This calls Rtree's native vector API
    directly because ``Index.intersection_v`` can pass a NumPy scalar to
    ``ndarray.resize`` when dense results outgrow its initial buffer.
    """
    # Normalize the bounds into the dtype, shape, and contiguous layout expected by
    # libspatialindex. Keeping these arrays alive also keeps the pointers below valid.
    minima, maxima = tree._prepare_v_arrays(bounds[:, :3], bounds[:, 3:])
    query_count, dimensions = minima.shape
    if query_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)

    # The C API walks the two-dimensional coordinate arrays using strides measured
    # in elements rather than bytes, hence the division by NumPy's item size.
    row_stride = minima.strides[0] // minima.itemsize
    column_stride = minima.strides[1] // minima.itemsize

    # Rtree's wrapper starts with room for two matches per query. ``intersection_ids``
    # grows when that estimate is too small; ``intersection_counts`` always has one
    # slot per query and tells us how the flat ID array is partitioned.
    intersection_ids = np.empty(2 * query_count, dtype=np.int64)
    intersection_counts = np.empty(query_count, dtype=np.uint64)

    # libspatialindex writes the number of fully completed queries through this pointer.
    processed_query_count = ctypes.c_int64(0)
    query_offset = 0
    result_offset = 0

    while True:
        # Resume at the first unfinished query and the first unused result slot. The
        # ``.ctypes.data`` values are raw addresses into the live NumPy allocations;
        # the native call does not retain them after returning.
        rtree_core.rt.Index_Intersects_id_v(
            tree.handle,
            query_count - query_offset,
            dimensions,
            len(intersection_ids) - result_offset,
            row_stride,
            column_stride,
            minima[query_offset:].ctypes.data,
            maxima[query_offset:].ctypes.data,
            intersection_ids[result_offset:].ctypes.data,
            intersection_counts[query_offset:].ctypes.data,
            ctypes.byref(processed_query_count),
        )
        if processed_query_count.value == query_count - query_offset:
            populated_result_count = int(intersection_counts.sum())
            return (
                intersection_ids[:populated_result_count],
                intersection_counts,
            )

        # A full output buffer may stop partway through the query list. Preserve the
        # IDs belonging to every fully completed query, then resume from the blocked
        # query. Its count slot contains the capacity required for that query.
        completed_slice = slice(
            query_offset, query_offset + processed_query_count.value
        )
        result_offset += int(intersection_counts[completed_slice].sum())
        query_offset += processed_query_count.value
        blocked_query_size = int(intersection_counts[query_offset])

        # Grow in place so completed IDs remain intact. The explicit ``int`` is the
        # dense-result bug fix: NumPy rejects Rtree's uint64 scalar as a resize shape.
        # ``refcheck=False`` is safe here because the native call retained no pointer.
        new_capacity = 2 * len(intersection_ids) + blocked_query_size
        intersection_ids.resize(new_capacity, refcheck=False)


@dataclass
class StepStats:
    """Record candidate counts and broad/narrow phase durations for one completed step."""

    step: int
    candidates: int
    collisions: int
    broad_s: float
    narrow_s: float

    @property
    def detection_s(self) -> float:
        """Return total collision-detection time for this step in seconds."""
        return self.broad_s + self.narrow_s


class TetrahedralCollisionSimulator:
    """Advance two meshes through collision detection and retain per-step results.

    The simulator owns geometry and detection state. Polyscope rendering is reached only
    through the visualization bridge at the end of the class; interactive playback is
    owned separately by :class:`PolyscopeSimulationUI`.
    """

    def __init__(
        self,
        config: SimulationConfig,
        detector: CollisionDetector,
        static_mesh: TetrahedralMesh,
        moving_mesh: TetrahedralMesh,
    ):
        """Initialize simulation state and acceleration data for the supplied meshes."""
        self.config = config
        self.detector = detector
        self.static_mesh = static_mesh
        self.moving_mesh = moving_mesh
        self.visualization_manager = VisualizationManager(config)
        self.step_size = -config.initial_offset / config.num_steps
        self.home_vertices = [mesh.vertices.copy() for mesh in self.meshes]
        self._refresh_static_acceleration()
        self.moving_connectivity_device = (
            torch.as_tensor(
                self.moving_mesh.tetrahedra,
                dtype=torch.long,
                device=self.detector.device,
            )
            if self.detector.is_cuda
            else None
        )
        self.current_step = 0
        self.candidate_pairs = _NO_PAIRS
        self.collision_pairs = _NO_PAIRS
        self.history: List[StepStats] = []
        self.run_started: Optional[float] = None

    @property
    def meshes(self) -> Tuple[TetrahedralMesh, TetrahedralMesh]:
        """Return the static and moving meshes in candidate-pair column order."""
        return self.static_mesh, self.moving_mesh

    # Collision detection

    def _refresh_static_acceleration(self) -> None:
        """Rebuild static bounds, its Rtree, and cached detector frames after movement."""
        self.static_mesh_bounds = np.concatenate(
            (
                self.static_mesh.vertices.min(axis=0),
                self.static_mesh.vertices.max(axis=0),
            )
        )
        self.static_bounds_tree = trimesh.util.bounds_tree(self.static_mesh.bounds)
        self.detector.prepare_static(
            self.static_mesh.vertices[self.static_mesh.tetrahedra]
        )

    def broad_phase(self) -> np.ndarray:
        """Return ``(n, 2)`` static/moving cell indices whose AABBs overlap."""
        moving_bounds = self.moving_mesh.bounds
        overlaps_static_bounds = np.all(
            moving_bounds[:, 3:] >= self.static_mesh_bounds[:3], axis=1
        )
        overlaps_static_bounds &= np.all(
            moving_bounds[:, :3] <= self.static_mesh_bounds[3:], axis=1
        )
        moving_query_indices = np.flatnonzero(overlaps_static_bounds)
        if len(moving_query_indices) == 0:
            return _NO_PAIRS

        if len(moving_query_indices) < 0.75 * len(moving_bounds):
            query_bounds = moving_bounds[moving_query_indices]
        else:
            query_bounds = moving_bounds
            moving_query_indices = np.arange(len(moving_bounds), dtype=np.int64)
        static_indices, matches_per_moving_tetrahedron = _bulk_tree_intersections(
            self.static_bounds_tree, query_bounds
        )
        moving_indices = np.repeat(
            moving_query_indices,
            matches_per_moving_tetrahedron.astype(np.intp, copy=False),
        )
        return np.column_stack((static_indices, moving_indices)).astype(
            np.int64, copy=False
        )

    def narrow_phase(self, candidates: np.ndarray) -> np.ndarray:
        """Return the subset of ``candidates`` that the model classifies as intersecting."""
        if len(candidates) == 0:
            return candidates
        if self.detector.is_cuda:
            return self._narrow_phase_cuda(candidates)

        static_candidate_indices = candidates[:, 0]
        moving_candidate_indices = candidates[:, 1]
        if self.detector.pca_unit_frames is not None:
            collision_mask = np.concatenate(
                [
                    self.detector.classify_pca_unit(
                        static_candidate_indices[
                            offset : offset + self.config.batch_size
                        ],
                        self.moving_mesh.vertices[
                            self.moving_mesh.tetrahedra[
                                moving_candidate_indices[
                                    offset : offset + self.config.batch_size
                                ]
                            ]
                        ],
                    )
                    for offset in range(0, len(candidates), self.config.batch_size)
                ]
            )
            return candidates[collision_mask]

        static_vertices = self.static_mesh.corners_of(static_candidate_indices)
        moving_vertices = self.moving_mesh.corners_of(moving_candidate_indices)
        pair_data = torch.from_numpy(
            np.concatenate([static_vertices, moving_vertices], axis=1)
        )
        collision_mask = np.concatenate(
            [
                self.detector.classify(
                    pair_data[offset : offset + self.config.batch_size]
                )
                for offset in range(0, len(candidates), self.config.batch_size)
            ]
        )
        return candidates[collision_mask]

    def _narrow_phase_cuda(self, candidates: np.ndarray) -> np.ndarray:
        """Classify candidates on CUDA and copy only the final collision mask to the host."""
        candidate_pairs_device = torch.as_tensor(
            candidates, dtype=torch.long, device=self.detector.device
        )
        hit_batches = []

        if self.detector.pca_unit_frames is not None:
            all_moving_vertices = torch.as_tensor(
                self.moving_mesh.vertices,
                dtype=self.detector.dtype,
                device=self.detector.device,
            )
            all_moving_tetrahedra = all_moving_vertices[self.moving_connectivity_device]

            for offset in range(0, len(candidates), self.config.batch_size):
                candidate_batch = candidate_pairs_device[
                    offset : offset + self.config.batch_size
                ]
                static_indices = candidate_batch[:, 0]
                moving_indices = candidate_batch[:, 1]
                moving_batch = all_moving_tetrahedra[moving_indices]
                hit_batches.append(
                    self.detector.classify_pca_unit_tensor(static_indices, moving_batch)
                )
        else:
            static_vertices = self.static_mesh.corners_of(candidates[:, 0])
            moving_vertices = self.moving_mesh.corners_of(candidates[:, 1])
            flattened_pairs = np.concatenate([static_vertices, moving_vertices], axis=1)
            pair_data_device = torch.as_tensor(
                flattened_pairs,
                dtype=self.detector.dtype,
                device=self.detector.device,
            )

            for offset in range(0, len(candidates), self.config.batch_size):
                pair_batch = pair_data_device[offset : offset + self.config.batch_size]
                hit_batches.append(self.detector.classify_tensor(pair_batch))

        collision_mask = torch.cat(hit_batches).cpu().numpy()
        return candidates[collision_mask]

    def detect_collisions(self) -> Tuple[float, float]:
        """Run both detection phases, store their pairs, and return phase durations."""
        start = time.perf_counter()
        self.candidate_pairs = self.broad_phase()
        broad_s = time.perf_counter() - start

        start = time.perf_counter()
        self.collision_pairs = self.narrow_phase(self.candidate_pairs)
        return broad_s, time.perf_counter() - start

    # Simulation lifecycle

    def step_forward(self) -> None:
        """Advance one frame, detect collisions, refresh rendering, and record timings."""
        if self.current_step >= self.config.num_steps:
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
            self._refresh_static_acceleration()

        broad_s, narrow_s = self.detect_collisions()
        self.refresh_visualization()
        self.current_step += 1
        self._log_step(broad_s, narrow_s)

        if self.current_step >= self.config.num_steps:
            self.log_summary()

    def reset_simulation(self) -> None:
        """Restore initial geometry and clear all candidate, collision, and timing state."""
        self.current_step = 0
        self.candidate_pairs = _NO_PAIRS
        self.collision_pairs = _NO_PAIRS
        self.history.clear()
        self.run_started = None
        for mesh, home in zip(self.meshes, self.home_vertices):
            mesh.set_vertices(home.copy())
        self._refresh_static_acceleration()
        self.refresh_visualization()
        logger.info("Simulation reset.")

    # Reporting

    def _log_step(self, broad_s: float, narrow_s: float) -> None:
        """Append one timing record and log its progress and aggregate throughput."""
        stats = StepStats(
            self.current_step,
            len(self.candidate_pairs),
            len(self.collision_pairs),
            broad_s,
            narrow_s,
        )
        self.history.append(stats)
        total_candidates = sum(s.candidates for s in self.history)
        detection_s = sum(s.detection_s for s in self.history)
        aggregate = (
            f"{detection_s / total_candidates * 1e6:6.2f}us/cand"
            if total_candidates
            else "-".center(13)
        )
        logger.info(
            "step %3d/%-3d  cand %11s  hit %11s  broad %7.3fs  narrow %7.3fs  elapsed %7.1fs  agg %s",
            stats.step,
            self.config.num_steps,
            f"{stats.candidates:,}",
            f"{stats.collisions:,}",
            stats.broad_s,
            stats.narrow_s,
            time.perf_counter() - self.run_started,
            aggregate,
        )

    def log_summary(self) -> None:
        """Log aggregate phase timings and identify the busiest and slowest steps."""
        if not self.history:
            return
        total_candidates = sum(s.candidates for s in self.history)
        total_collisions = sum(s.collisions for s in self.history)
        detection_s = sum(s.detection_s for s in self.history)
        busiest = max(self.history, key=lambda s: s.candidates)
        slowest = max(self.history, key=lambda s: s.detection_s)

        logger.info("")
        logger.info(
            "run summary   %d steps  %s candidates  %s collisions",
            len(self.history),
            f"{total_candidates:,}",
            f"{total_collisions:,}",
        )
        for label, attr in (("broad", "broad_s"), ("narrow", "narrow_s")):
            get = attrgetter(attr)
            times = [get(s) for s in self.history]
            peak = max(self.history, key=get)
            logger.info(
                "  %-6s      %8.2fs total  %7.3fs mean  %7.3fs peak (step %d)",
                label,
                sum(times),
                sum(times) / len(times),
                get(peak),
                peak.step,
            )
        broad_share = (
            100.0 * sum(s.broad_s for s in self.history) / detection_s
            if detection_s
            else 0.0
        )
        logger.info(
            "  detection   %8.2fs total  %s  broad %.0f%% / narrow %.0f%%",
            detection_s,
            f"{detection_s / total_candidates * 1e6:.2f}us/cand aggregate"
            if total_candidates
            else "no candidates",
            broad_share,
            100.0 - broad_share,
        )
        logger.info(
            "  busiest     step %d (%s candidates)   slowest  step %d (%.2fs)",
            busiest.step,
            f"{busiest.candidates:,}",
            slowest.step,
            slowest.detection_s,
        )

    # Visualization bridge

    def initialize_visualization(self, force_headless: bool = False) -> None:
        """Initialize Polyscope and register both meshes for subsequent frame updates."""
        self.visualization_manager.initialize(force_headless)
        for mesh in self.meshes:
            self.visualization_manager.register(mesh)

    def refresh_visualization(self) -> None:
        """Push current vertices and per-cell collision colors to Polyscope."""
        for column, mesh in enumerate(self.meshes):
            self.visualization_manager.sync_positions(mesh)
            self.visualization_manager.shade_collisions(
                mesh, self.collision_pairs[:, column]
            )


# --------------------------------------------------------------------------- #
# Interactive UI
# --------------------------------------------------------------------------- #


class PolyscopeSimulationUI:
    """Own the interactive playback state and render Polyscope's ImGui controls."""

    def __init__(self, simulator: TetrahedralCollisionSimulator):
        """Attach controls to ``simulator`` with playback initially paused."""
        self.simulator = simulator
        self.is_playing = False

    def draw(self) -> None:
        """Draw one control-panel frame and advance the simulation during playback."""
        simulator = self.simulator
        config = simulator.config

        psim.Text(f"Step: {simulator.current_step}/{config.num_steps}")
        psim.Text(f"Transforms: {' -> '.join(simulator.detector.transforms)}")
        psim.Text(
            f"Inference: {simulator.detector.device} / "
            f"{str(simulator.detector.dtype).removeprefix('torch.')} / "
            f"batch {config.batch_size:,}"
        )

        if self.is_playing:
            if psim.Button("Stop"):
                self.is_playing = False
        elif psim.Button("Play") and simulator.current_step < config.num_steps:
            self.is_playing = True

        psim.SameLine()
        if psim.Button("Reset"):
            self.is_playing = False
            simulator.reset_simulation()

        psim.Separator()
        psim.Text(f"Candidate Pairs: {len(simulator.candidate_pairs)}")
        psim.Text(f"Colliding Pairs: {len(simulator.collision_pairs)}")

        if self.is_playing:
            if simulator.current_step < config.num_steps:
                simulator.step_forward()
            else:
                self.is_playing = False


STATIC_COLOR = [0.3, 0.3, 1.0]
MOVING_COLOR = [0.3, 1.0, 0.3]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse simulator options from ``argv`` or, when omitted, from ``sys.argv``."""
    parser = argparse.ArgumentParser(
        description="Run the tetrahedral collision simulator"
    )

    inputs = parser.add_argument_group("model and meshes")
    inputs.add_argument(
        "--model",
        "-M",
        default="model/EuroGraphicsDemo_PCA_Unit.pt",
        help="Path to the TorchScript collision model",
    )
    inputs.add_argument(
        "--transform",
        "-T",
        nargs="+",
        choices=sorted(TRANSFORMS),
        metavar="NAME",
        help=(
            "Transform chain applied before inference, in order "
            f"(choices: {sorted(TRANSFORMS)}); defaults to the chain registered "
            "for the selected model"
        ),
    )
    inputs.add_argument(
        "--mesh-a",
        default="data/armadillo.obj",
        help="Path to the first (static) mesh OBJ file",
    )
    inputs.add_argument(
        "--mesh-b",
        help=(
            "Path to the second (moving) mesh OBJ file; defaults to the same file "
            "as --mesh-a"
        ),
    )

    motion = parser.add_argument_group("simulation")
    motion.add_argument(
        "--num-steps", type=int, default=50, help="Number of simulation steps"
    )
    motion.add_argument(
        "--offset-x",
        type=float,
        default=2.5,
        help="Initial x offset for the moving mesh",
    )
    motion.add_argument(
        "--rotate-static",
        action="store_true",
        help="Rotate the static mesh while the second mesh moves",
    )
    motion.add_argument(
        "--rotation-speed",
        type=float,
        default=np.radians(10.0),
        help="Static mesh rotation speed in radians per step",
    )
    motion.add_argument(
        "--rotation-axis",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        help="Rotation axis for the static mesh",
    )

    inference = parser.add_argument_group("inference")
    inference.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Tetrahedron pairs per inference batch "
            "(default: 65,536 on CUDA; 2,048 on CPU)"
        ),
    )
    inference.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available",
    )
    precision = inference.add_mutually_exclusive_group()
    precision.add_argument(
        "--float32",
        action="store_true",
        help="Force float32 transforms and inference (the CUDA default)",
    )
    precision.add_argument(
        "--float64",
        action="store_true",
        help="Force model-native float64 transforms and inference (the CPU default)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--headless",
        action="store_true",
        help="Skip the interactive window and render every step to PNG screenshots",
    )
    output.add_argument(
        "--screenshot-dir",
        default="screenshots",
        help="Directory for headless screenshots",
    )
    output.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="Logging verbosity; DEBUG adds timestamps and reveals Polyscope's banner",
    )
    return parser.parse_args(argv)


def main():
    """Build the configured simulator and run either its headless or interactive UI."""
    runtime = RuntimeConfig.from_args(parse_args())
    configure_logging(runtime.log_level)

    detector = CollisionDetector(
        runtime.model_path,
        runtime.transform_names,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    logger.info("Inference batch size: %s pairs", f"{runtime.simulation.batch_size:,}")

    static_mesh = TetrahedralMesh.from_file(
        runtime.static_mesh_path, "Mesh A", STATIC_COLOR
    )
    if (
        Path(runtime.moving_mesh_path).resolve()
        == Path(runtime.static_mesh_path).resolve()
    ):
        moving_mesh = TetrahedralMesh(
            "Mesh B",
            MOVING_COLOR,
            static_mesh.vertices.copy() + runtime.simulation.initial_offset,
            static_mesh.tetrahedra.copy(),
        )
        logger.info(
            "Mesh B reused Mesh A's tetrahedralization (%s tetrahedra)",
            f"{len(moving_mesh.tetrahedra):,}",
        )
    else:
        moving_mesh = TetrahedralMesh.from_file(
            runtime.moving_mesh_path,
            "Mesh B",
            MOVING_COLOR,
            offset=runtime.simulation.initial_offset,
        )

    simulator = TetrahedralCollisionSimulator(
        runtime.simulation, detector, static_mesh, moving_mesh
    )
    simulator.initialize_visualization(force_headless=runtime.headless)

    if simulator.visualization_manager.headless:
        out_dir = runtime.screenshot_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ps.screenshot(str(out_dir / "step_000.png"))
        for _ in range(runtime.simulation.num_steps):
            simulator.step_forward()
            ps.screenshot(str(out_dir / f"step_{simulator.current_step:03d}.png"))
        logger.info(
            "Wrote %d frames to %s",
            runtime.simulation.num_steps + 1,
            out_dir.resolve(),
        )
        return

    simulation_ui = PolyscopeSimulationUI(simulator)
    ps.set_user_callback(simulation_ui.draw)
    ps.show()


if __name__ == "__main__":
    main()
