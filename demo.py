"""
Tetrahedral Collision Simulator

A modular simulation system for detecting collisions between tetrahedral meshes
using PyTorch models for narrow-phase collision detection.
This version implements a two-stage detection pipeline:
1. Broad Phase: A highly efficient R-tree implementation using trimesh.util.bounds_tree.
2. Narrow Phase: Vectorized data gathering and precise model-based checking with timing.
3. Visualization: Renders full tetrahedral volumes to highlight colliding cells.
"""

import trimesh
import trimesh.util
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import torch
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging
import time
import threading
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    num_steps: int = 50
    initial_offset: np.ndarray = None
    default_color_a: List[float] = None
    default_color_b: List[float] = None
    collision_color: List[float] = None
    batch_size: int = 2048

    def __post_init__(self):
        if self.initial_offset is None: self.initial_offset = np.array([3.0, 0.0, 0.0])
        if self.default_color_a is None: self.default_color_a = [0.3, 0.3, 1.0]  # Blue
        if self.default_color_b is None: self.default_color_b = [0.3, 1.0, 0.3]  # Green
        if self.collision_color is None: self.collision_color = [1.0, 0.0, 0.0]  # Red

@dataclass
class TetrahedralMesh:
    """Container for tetrahedral mesh data."""
    vertices: np.ndarray
    tetrahedra: np.ndarray
    original_mesh: trimesh.Trimesh
    aabbs: np.ndarray


class Preprocessing:
    """Handles vectorized data transformations for the PyTorch model."""
    @staticmethod
    def principal_axis_transform(input_tensor: torch.Tensor) -> torch.Tensor:
        data = input_tensor.to(torch.float64)
        batch_size = data.size(0)
        tetra1 = data[:, :12].view(batch_size, 4, 3)
        tetra2 = data[:, 12:].view(batch_size, 4, 3)
        centroid1 = torch.mean(tetra1, dim=1, keepdim=True)
        tetra1_centered = tetra1 - centroid1
        cov_matrix = torch.bmm(tetra1_centered.transpose(1, 2), tetra1_centered)
        _, eigenvectors = torch.linalg.eigh(cov_matrix)
        tetra2_centered = tetra2 - centroid1
        tetra1_transformed = torch.bmm(tetra1_centered, eigenvectors)
        tetra2_transformed = torch.bmm(tetra2_centered, eigenvectors)
        return torch.cat([
            tetra1_transformed.view(batch_size, 12),
            tetra2_transformed.view(batch_size, 12)
        ], dim=1).to(torch.float64)

class TetrahedralGenerator:
    """Handles tetrahedralization and property calculation for meshes."""
    @staticmethod
    def tetrahedralize_with_tetgen(mesh: trimesh.Trimesh) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            import tetgen
            tet = tetgen.TetGen(mesh.vertices, mesh.faces)
            vertices, tetrahedra = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, quality=True)
            logger.info(f"Generated {len(tetrahedra)} tetrahedra from {len(vertices)} vertices")
            return vertices, tetrahedra
        except Exception as e:
            logger.error(f"TetGen tetrahedralization failed: {e}")
            return None, None

    @staticmethod
    def calculate_aabbs(vertices: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
        tet_vertices = vertices[tetrahedra]
        return np.stack([np.min(tet_vertices, axis=1), np.max(tet_vertices, axis=1)], axis=1)

class CollisionDetector:
    """Handles collision detection between tetrahedra using a PyTorch model."""
    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.model = model
        self.device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model: self.set_model(model)

    def set_model(self, model: torch.nn.Module):
        self.model = model
        self.model.to(self.device).eval()
        logger.info(f"PyTorch collision model loaded onto {self.device}!")

    def detect_collisions_in_batch(self, batch_tensor: torch.Tensor) -> List[bool]:
        if not self.model: return [False] * len(batch_tensor)
        try:
            with torch.no_grad():
                input_tensor = batch_tensor.to(self.device)
                transformed = Preprocessing.principal_axis_transform(input_tensor)
                output = self.model(transformed)
                return (output > 0.5).cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"PyTorch model inference failed: {e}")
            return [False] * len(batch_tensor)


class VisualizationManager:
    """Manages Polyscope volume mesh visualization."""
    def __init__(self, config: SimulationConfig): self.config = config
    def initialize(self): ps.init()
    
    def register_mesh(self, name: str, mesh: TetrahedralMesh, color: List[float]):
        ps_mesh = ps.register_volume_mesh(name, mesh.vertices, mesh.tetrahedra)
        ps_mesh.set_color(color)

    def update_mesh_positions(self, name: str, vertices: np.ndarray):
        ps.get_volume_mesh(name).update_vertex_positions(vertices)
    
    def update_collision_colors(self, name_a: str, mesh_a: TetrahedralMesh, name_b: str, mesh_b: TetrahedralMesh, collision_pairs: List[Tuple[int, int]]):
        colors_a = np.tile(self.config.default_color_a, (len(mesh_a.tetrahedra), 1))
        colors_b = np.tile(self.config.default_color_b, (len(mesh_b.tetrahedra), 1))
        colliding_tets_a = {p[0] for p in collision_pairs}
        colliding_tets_b = {p[1] for p in collision_pairs}
        for tet_idx in colliding_tets_a: colors_a[tet_idx] = self.config.collision_color
        for tet_idx in colliding_tets_b: colors_b[tet_idx] = self.config.collision_color
        ps.get_volume_mesh(name_a).add_color_quantity("collisions", colors_a, defined_on='cells', enabled=True)
        ps.get_volume_mesh(name_b).add_color_quantity("collisions", colors_b, defined_on='cells', enabled=True)


class TetrahedralCollisionSimulator:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.collision_detector = CollisionDetector()
        self.visualization_manager = VisualizationManager(self.config)
        self.mesh_a: Optional[TetrahedralMesh] = None
        self.mesh_b: Optional[TetrahedralMesh] = None
        self.current_step = 0
        self.step_size: Optional[np.ndarray] = None
        self.collision_pairs: List[Tuple[int, int]] = []
        self.candidate_pairs: List[Tuple[int, int]] = []  # Pairs of indices for broad phase
        self.is_playing: bool = False  # State for play/stop functionality
        self._inference_lock = threading.Lock()
        
    def set_collision_model(self, model: torch.nn.Module):
        self.collision_detector.set_model(model)
    
    def load_mesh(self, mesh_path: str, apply_offset: bool = False) -> Optional[TetrahedralMesh]:
        try:
            original_mesh = trimesh.load(mesh_path, force='mesh')
            if apply_offset: original_mesh.apply_translation(self.config.initial_offset)
            vertices, tetrahedra = TetrahedralGenerator.tetrahedralize_with_tetgen(original_mesh)
            if vertices is None: return None
            aabbs = TetrahedralGenerator.calculate_aabbs(vertices, tetrahedra)
            return TetrahedralMesh(vertices, tetrahedra, original_mesh, aabbs)
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_path}: {e}")
            return None
    
    def load_meshes(self, mesh_a_path: str, mesh_b_path: str) -> bool:
        self.mesh_a = self.load_mesh(mesh_a_path)
        self.mesh_b = self.load_mesh(mesh_b_path, apply_offset=True)
        if not (self.mesh_a and self.mesh_b): return False
        self.step_size = -self.config.initial_offset / self.config.num_steps
        logger.info("Successfully loaded and prepared meshes.")
        return True
    

    def broad_phase(self):

        broad_phase_start_time = time.time()

        # Debug: Check the relationship between AABBs and tetrahedra
        logger.info(f"Mesh A: {len(self.mesh_a.tetrahedra)} tetrahedra, {len(self.mesh_a.aabbs)} AABBs")
        logger.info(f"Mesh B: {len(self.mesh_b.tetrahedra)} tetrahedra, {len(self.mesh_b.aabbs)} AABBs")

        # Ensure we only use AABBs that correspond to actual tetrahedra
        max_tetrahedra_a = len(self.mesh_a.tetrahedra)
        max_tetrahedra_b = len(self.mesh_b.tetrahedra)

        # 1. Create an R-tree for the AABBs of the first mesh.
        bounds_a = self.mesh_a.aabbs[:max_tetrahedra_a].reshape(-1, 6)
        tree_a = trimesh.util.bounds_tree(bounds_a)

        # 2. Use a set to store unique pairs and avoid duplicates
        candidate_pairs_set = set()
        bounds_b = self.mesh_b.aabbs[:max_tetrahedra_b].reshape(-1, 6)

        for index_b, aabb_b in enumerate(bounds_b):
            # Safety check: ensure index_b is within bounds
            if index_b >= max_tetrahedra_b:
                continue
                
            # The 'intersection' method returns indices from tree_a (mesh_a)
            intersecting_indices_a = tree_a.intersection(aabb_b)
            
            for index_a in intersecting_indices_a:
                # Safety check: ensure index_a is within bounds
                if index_a >= max_tetrahedra_a:
                    continue
                candidate_pairs_set.add((index_a, index_b))

        # Convert set back to list for compatibility with existing code
        self.candidate_pairs = list(candidate_pairs_set)

        logger.info(f"Broad phase complete in {time.time() - broad_phase_start_time:.4f}s. Found {len(self.candidate_pairs)} unique candidates.")

    def narrow_phase(self):
        """Perform narrow-phase collision detection in a single-threaded loop."""
        start_time = time.time()

        # No candidates, clear and exit
        if not self.candidate_pairs:
            self.collision_pairs = []
            return

        # Prepare vertex data for all candidate pairs
        pairs = np.array(self.candidate_pairs)
        idx_a, idx_b = pairs[:, 0], pairs[:, 1]
        verts_a = self.mesh_a.vertices[self.mesh_a.tetrahedra[idx_a]].reshape(-1, 12)
        verts_b = self.mesh_b.vertices[self.mesh_b.tetrahedra[idx_b]].reshape(-1, 12)
        data = np.concatenate((verts_a, verts_b), axis=1)

        # Batch inference loop (single-threaded)
        self.collision_pairs = []
        batch_size = self.config.batch_size
        total = len(self.candidate_pairs)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_data = torch.tensor(data[start:end], dtype=torch.float64)
            results = self.collision_detector.detect_collisions_in_batch(batch_data)
            for pair, hit in zip(self.candidate_pairs[start:end], results):
                if hit:
                    self.collision_pairs.append(pair)

        elapsed = time.time() - start_time
        logger.info(f"Narrow phase complete in {elapsed:.4f}s. Confirmed {len(self.collision_pairs)} colliding pairs.")


    def detect_collisions(self):
        if not self.mesh_a or not self.mesh_b: return

        self.broad_phase()

        # Debug: Check the relationship between AABBs and tetrahedra
        logger.info(f"Mesh A: {len(self.mesh_a.tetrahedra)} tetrahedra, {len(self.mesh_a.aabbs)} AABBs")
        logger.info(f"Mesh B: {len(self.mesh_b.tetrahedra)} tetrahedra, {len(self.mesh_b.aabbs)} AABBs")

        if not self.candidate_pairs:
            self.collision_pairs = []
            return

        self.narrow_phase()

    def step_forward(self):
        if self.current_step >= self.config.num_steps:
            self.is_playing = False # Stop playing at the end
            return
        if not self.mesh_b or self.step_size is None: return
        
        self.mesh_b.vertices += self.step_size
        self.mesh_b.aabbs = TetrahedralGenerator.calculate_aabbs(self.mesh_b.vertices, self.mesh_b.tetrahedra)
        self.visualization_manager.update_mesh_positions("Mesh B", self.mesh_b.vertices)
        
        self.detect_collisions()
        self.visualization_manager.update_collision_colors("Mesh A", self.mesh_a, "Mesh B", self.mesh_b, self.collision_pairs)
        self.current_step += 1
    
    def reset_simulation(self):
        self.is_playing = False # Stop playing on reset
        if not self.mesh_b or self.step_size is None: return
        
        self.mesh_b.vertices -= self.step_size * self.current_step
        self.mesh_b.aabbs = TetrahedralGenerator.calculate_aabbs(self.mesh_b.vertices, self.mesh_b.tetrahedra)
        self.visualization_manager.update_mesh_positions("Mesh B", self.mesh_b.vertices)
        
        self.current_step = 0
        self.collision_pairs = []
        self.visualization_manager.update_collision_colors("Mesh A", self.mesh_a, "Mesh B", self.mesh_b, [])
        logger.info("Simulation reset.")
    
    def initialize_visualization(self):
        self.visualization_manager.initialize()
        if self.mesh_a and self.mesh_b:
            self.visualization_manager.register_mesh("Mesh A", self.mesh_a, self.config.default_color_a)
            self.visualization_manager.register_mesh("Mesh B", self.mesh_b, self.config.default_color_b)
    
    def create_ui_callback(self):
        def ui_callback():
            psim.Text(f"Step: {self.current_step}/{self.config.num_steps}")

            # Show Play or Stop button based on the current state and update the state
            if not self.is_playing:
                if psim.Button("Play"):
                    # Start playing if not already at the end
                    if self.current_step < self.config.num_steps:
                        self.is_playing = True
            else:
                if psim.Button("Stop"):
                    self.is_playing = False

            psim.SameLine()
            if psim.Button("Reset"): 
                self.reset_simulation() # This method also sets is_playing to False
            
            psim.Separator()
            psim.Text(f"Candidate Pairs: {len(self.candidate_pairs)}")
            psim.Text(f"Colliding Pairs: {len(self.collision_pairs)}")

            # If playing, advance the simulation for the NEXT frame
            if self.is_playing:
                self.step_forward()
            
        return ui_callback


def main():
    config = SimulationConfig(num_steps=50, initial_offset=np.array([2.5, 0.0, 0.0]))
    simulator = TetrahedralCollisionSimulator(config)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load('model/TetrahedronPairNet_L_principal_axis.pt', map_location=device)
        model.double()
        simulator.set_collision_model(model)
    except FileNotFoundError:
        logger.error("Model file not found. Place it at 'model/TetrahedronPairNet_L_principal_axis.pt'")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model: {e}")

    if not simulator.load_meshes("data/mug.obj", "data/soccerball.obj"):
        return
    
    simulator.initialize_visualization()
    ps.set_user_callback(simulator.create_ui_callback())
    ps.show()


if __name__ == "__main__":
    main()