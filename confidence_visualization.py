"""
Tetrahedron Collision Confidence Visualization

Visualizes neural network collision detection confidence as one tetrahedron 
moves through another. Demonstrates the model's discrimination quality.
"""

import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

COLLISION_THRESHOLD = 0.5
WIREFRAME_RADIUS = 0.02

COLOR_STATIONARY = (0.2, 0.5, 0.9)
COLOR_COLLISION = (1.0, 0.2, 0.2)
COLOR_NO_COLLISION = (0.2, 0.8, 0.2)

TETRAHEDRON_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VisualizationConfig:
    num_positions: int = 100
    x_range: Tuple[float, float] = (-3.0, 3.0)
    plot_height: int = 200
    plot_width: int = 800


# =============================================================================
# Model Preprocessing
# =============================================================================

class Preprocessing:
    """Aligns tetrahedra to principal axes before model inference."""
    
    @staticmethod
    def principal_axis_transform(input_tensor: torch.Tensor) -> torch.Tensor:
        data = input_tensor.to(torch.float64)
        batch_size = data.size(0)
        
        tetra1 = data[:, :12].view(batch_size, 4, 3)
        tetra2 = data[:, 12:].view(batch_size, 4, 3)
        
        centroid1 = torch.mean(tetra1, dim=1, keepdim=True)
        tetra1_centered = tetra1 - centroid1
        tetra2_centered = tetra2 - centroid1
        
        cov_matrix = torch.bmm(tetra1_centered.transpose(1, 2), tetra1_centered)
        _, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        tetra1_transformed = torch.bmm(tetra1_centered, eigenvectors)
        tetra2_transformed = torch.bmm(tetra2_centered, eigenvectors)
        
        return torch.cat([
            tetra1_transformed.view(batch_size, 12),
            tetra2_transformed.view(batch_size, 12)
        ], dim=1).to(torch.float64)


# =============================================================================
# Tetrahedron I/O
# =============================================================================

def load_tetrahedron_from_obj(filepath: str) -> np.ndarray:
    """Load tetrahedron vertices from OBJ file. Returns (4, 3) array."""
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    if len(vertices) != 4:
        raise ValueError(f"Expected 4 vertices, got {len(vertices)}")
    
    return vertices


def create_default_tetrahedron() -> np.ndarray:
    """Create a regular tetrahedron centered at origin."""
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0]
    ], dtype=np.float64)


# =============================================================================
# Main Visualizer
# =============================================================================

class ConfidenceVisualizer:
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tet_stationary: Optional[np.ndarray] = None
        self.tet_moving_base: Optional[np.ndarray] = None
        
        self.x_positions: np.ndarray = np.array([])
        self.current_step: int = 0
        self.confidence_history: List[float] = []
    
    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    
    def load_model(self, model_path: str) -> bool:
        try:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.double()
            self.model.to(self.device).eval()
            logger.info(f"Model loaded onto {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_tetrahedra(self, stationary_path: Optional[str] = None, 
                        moving_path: Optional[str] = None) -> bool:
        try:
            if stationary_path:
                self.tet_stationary = load_tetrahedron_from_obj(stationary_path)
                logger.info(f"Loaded stationary tetrahedron from {stationary_path}")
            else:
                self.tet_stationary = create_default_tetrahedron()
                logger.info("Using default stationary tetrahedron")
            
            if moving_path:
                self.tet_moving_base = load_tetrahedron_from_obj(moving_path)
                logger.info(f"Loaded moving tetrahedron from {moving_path}")
            else:
                self.tet_moving_base = create_default_tetrahedron()
                logger.info("Using default moving tetrahedron")
            
            self.tet_stationary -= self.tet_stationary.mean(axis=0)
            self.tet_moving_base -= self.tet_moving_base.mean(axis=0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load tetrahedra: {e}")
            return False
    
    def initialize_steps(self):
        x_min, x_max = self.config.x_range
        self.x_positions = np.linspace(x_min, x_max, self.config.num_positions)
        self.confidence_history = []
        self.current_step = 0
        logger.info(f"Initialized {self.config.num_positions} steps from x={x_min:.2f} to x={x_max:.2f}")
    
    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    
    def _normalize_to_unit_cube(self, tet1: np.ndarray, tet2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale both tetrahedra uniformly to fit within unit cube."""
        combined = np.concatenate([tet1, tet2], axis=0)
        min_vals = combined.min(axis=0)
        max_range = (combined.max(axis=0) - min_vals).max() or 1.0
        
        return (tet1 - min_vals) / max_range, (tet2 - min_vals) / max_range
    
    def _get_moved_tetrahedron(self, step: int) -> np.ndarray:
        tet_moved = self.tet_moving_base.copy()
        tet_moved[:, 0] += self.x_positions[step]
        return tet_moved
    
    def compute_confidence(self, tet1: np.ndarray, tet2: np.ndarray) -> float:
        """Run model inference on a tetrahedron pair."""
        if self.model is None:
            return 0.0
        
        tet1_norm, tet2_norm = self._normalize_to_unit_cube(tet1, tet2)
        data = np.concatenate([tet1_norm.flatten(), tet2_norm.flatten()])
        tensor = torch.tensor(data, dtype=torch.float64).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            transformed = Preprocessing.principal_axis_transform(tensor)
            output = self.model(transformed)
            confidence = torch.sigmoid(output).cpu().numpy().flatten()[0]
        
        return float(confidence)
    
    def compute_and_record_confidence(self) -> float:
        """Compute confidence for current step and append to history."""
        tet_moved = self._get_moved_tetrahedron(self.current_step)
        conf = self.compute_confidence(self.tet_stationary, tet_moved)
        self.confidence_history.append(conf)
        return conf
    
    # -------------------------------------------------------------------------
    # Polyscope Visualization
    # -------------------------------------------------------------------------
    
    def initialize_polyscope(self):
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        
        ps_stationary = ps.register_curve_network(
            "Stationary Tetrahedron", self.tet_stationary, TETRAHEDRON_EDGES)
        ps_stationary.set_color(COLOR_STATIONARY)
        ps_stationary.set_radius(WIREFRAME_RADIUS)
        
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron", self._get_moved_tetrahedron(self.current_step), TETRAHEDRON_EDGES)
        ps_moving.set_color(COLOR_NO_COLLISION)
        ps_moving.set_radius(WIREFRAME_RADIUS)
        
        logger.info("Polyscope visualization initialized")
    
    def update_visualization(self, confidence: float):
        is_collision = confidence > COLLISION_THRESHOLD
        color = COLOR_COLLISION if is_collision else COLOR_NO_COLLISION
        
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron", self._get_moved_tetrahedron(self.current_step), TETRAHEDRON_EDGES)
        ps_moving.set_color(color)
        ps_moving.set_radius(WIREFRAME_RADIUS)
    
    # -------------------------------------------------------------------------
    # UI Components
    # -------------------------------------------------------------------------
    
    def _render_status_panel(self, confidence: float, x_position: float):
        is_collision = confidence > COLLISION_THRESHOLD
        
        color_red = (1.0, 0.4, 0.4, 1.0)
        color_green = (0.4, 1.0, 0.4, 1.0)
        color_info = (0.7, 0.85, 1.0, 1.0)
        
        psim.Text("Step")
        psim.SameLine()
        psim.TextColored(color_info, f"{self.current_step + 1} / {self.config.num_positions}")
        
        psim.Text("Position")
        psim.SameLine()
        psim.TextColored(color_info, f"x = {x_position:+.3f}")
        
        psim.Text("Confidence")
        psim.SameLine()
        psim.TextColored(color_red if is_collision else color_green, f"{confidence:.4f}")
        
        psim.Separator()
        
        if is_collision:
            psim.TextColored((1.0, 0.35, 0.35, 1.0), ">> INTERSECTION DETECTED <<")
        else:
            psim.TextColored((0.35, 0.9, 0.35, 1.0), "No Intersection")
    
    def _render_confidence_graph(self):
        psim.TextColored((0.8, 0.8, 0.8, 1.0), "Confidence Curve")
        
        if self.confidence_history:
            psim.PlotLines(
                "##confidence_plot",
                self.confidence_history.copy(),
                graph_size=(400, 140),
                scale_min=0.0,
                scale_max=1.0
            )
            psim.TextColored((0.6, 0.6, 0.6, 1.0), f"Inferences: {len(self.confidence_history)}")
        else:
            psim.TextColored((0.5, 0.5, 0.5, 1.0), "Move slider to build curve...")
        
        psim.TextColored((0.5, 0.5, 0.5, 1.0), f"Threshold: {COLLISION_THRESHOLD}")
    
    def _render_confidence_bar(self, confidence: float):
        bar_width = 50
        filled = int(confidence * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        color = (1.0, 0.4, 0.4, 1.0) if confidence > COLLISION_THRESHOLD else (0.4, 0.9, 0.4, 1.0)
        psim.TextColored(color, f"{bar} {confidence:.1%}")
    
    def create_ui_callback(self):
        def ui_callback():
            changed, new_step = psim.SliderInt(
                "##step_slider", self.current_step, 0, self.config.num_positions - 1)
            
            if changed and new_step != self.current_step:
                self.current_step = new_step
            
            current_conf = self.compute_and_record_confidence()
            current_x = self.x_positions[self.current_step]
            
            self.update_visualization(current_conf)
            
            psim.TextColored((0.9, 0.9, 0.9, 1.0), "COLLISION DETECTION")
            psim.Separator()
            
            self._render_status_panel(current_conf, current_x)
            psim.Separator()
            
            if psim.Button("Reset"):
                self.initialize_steps()
            
            psim.Separator()
            self._render_confidence_graph()
            psim.Separator()
            self._render_confidence_bar(current_conf)
        
        return ui_callback
    
    def run(self):
        if self.tet_stationary is None or self.tet_moving_base is None:
            logger.error("Tetrahedra not loaded")
            return
        
        if self.model is None:
            logger.warning("No model loaded - confidence values will be 0")
        
        self.initialize_steps()
        self.initialize_polyscope()
        ps.set_user_callback(self.create_ui_callback())
        ps.show()
    
    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    
    def compute_all_confidences_for_export(self):
        logger.info("Computing all confidence values for export...")
        self.initialize_steps()
        
        for step in range(self.config.num_positions):
            self.current_step = step
            tet_moved = self._get_moved_tetrahedron(step)
            conf = self.compute_confidence(self.tet_stationary, tet_moved)
            self.confidence_history.append(conf)
        
        logger.info(f"Computed {len(self.confidence_history)} values")
        logger.info(f"Range: [{min(self.confidence_history):.3f}, {max(self.confidence_history):.3f}]")
    
    def export_confidence_plot(self, output_path: str = "confidence_curve.png", 
                                figsize: Tuple[float, float] = (12, 4)):
        self.compute_all_confidences_for_export()
        
        fig, ax = plt.subplots(figsize=figsize)
        conf_array = np.array(self.confidence_history)
        
        # Color curve by collision state
        for i in range(len(self.x_positions) - 1):
            x = [self.x_positions[i], self.x_positions[i + 1]]
            y = [conf_array[i], conf_array[i + 1]]
            color = 'red' if conf_array[i] > COLLISION_THRESHOLD else 'green'
            ax.plot(x, y, color=color, linewidth=2)
        
        # Fill regions
        above = conf_array > COLLISION_THRESHOLD
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=above, alpha=0.3, color='red', label='Collision')
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=~above, alpha=0.3, color='green', label='No collision')
        
        ax.axhline(y=COLLISION_THRESHOLD, color='gray', linestyle='--', 
                   linewidth=1.5, label=f'Threshold ({COLLISION_THRESHOLD})')
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        ax.set_title('Collision Detection Confidence', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(self.config.x_range[0], self.config.x_range[1])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Annotate peak
        max_idx = np.argmax(conf_array)
        ax.annotate(f'Peak: {conf_array[max_idx]:.2f}', 
                    xy=(self.x_positions[max_idx], conf_array[max_idx]), 
                    xytext=(self.x_positions[max_idx] + 0.5, conf_array[max_idx] + 0.1),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
        plt.close()
        
        return output_path


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize collision detection confidence")
    parser.add_argument("--stationary", "-s", type=str, default=None,
                        help="Stationary tetrahedron OBJ file")
    parser.add_argument("--moving", "-m", type=str, default=None,
                        help="Moving tetrahedron OBJ file")
    parser.add_argument("--model", "-M", type=str, default="model/EuroGraphicsPaper.pt",
                        help="PyTorch model file")
    parser.add_argument("--positions", "-n", type=int, default=100,
                        help="Number of positions along trajectory")
    parser.add_argument("--x-min", type=float, default=-3.0)
    parser.add_argument("--x-max", type=float, default=3.0)
    parser.add_argument("--export", "-e", type=str, default=None,
                        help="Export plot to file (skips interactive mode)")
    parser.add_argument("--export-and-show", action="store_true",
                        help="Export plot and show interactive visualization")
    
    args = parser.parse_args()
    
    config = VisualizationConfig(
        num_positions=args.positions,
        x_range=(args.x_min, args.x_max)
    )
    
    visualizer = ConfidenceVisualizer(config)
    visualizer.load_model(args.model)
    
    if not visualizer.load_tetrahedra(args.stationary, args.moving):
        logger.error("Failed to load tetrahedra")
        return
    
    if args.export:
        visualizer.export_confidence_plot(args.export)
        if not args.export_and_show:
            return
    elif args.export_and_show:
        visualizer.export_confidence_plot("confidence_curve.png")
    
    visualizer.run()


if __name__ == "__main__":
    main()
