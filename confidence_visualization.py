"""
Tetrahedron Collision Confidence Visualization

This script visualizes the neural network's collision detection confidence
as one tetrahedron moves through another stationary tetrahedron.

The visualization shows:
- Top: Two wireframe tetrahedra (one stationary, one moving)
- Bottom: A confidence curve showing the model's output over the trajectory

This demonstrates the discrimination quality of the network.
"""

import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Collision detection threshold
COLLISION_THRESHOLD = 0.5

# Tetrahedron edge radius for visualization
WIREFRAME_RADIUS = 0.02

# Colors (RGB)
COLOR_STATIONARY = (0.2, 0.5, 0.9)    # Blue
COLOR_COLLISION = (1.0, 0.2, 0.2)      # Red
COLOR_NO_COLLISION = (0.2, 0.8, 0.2)   # Green

# Tetrahedron edges (6 edges connecting 4 vertices)
TETRAHEDRON_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


@dataclass
class VisualizationConfig:
    """Configuration for the confidence visualization."""
    num_positions: int = 100  # Number of positions along the trajectory
    x_range: Tuple[float, float] = (-3.0, 3.0)  # Movement range along X-axis
    plot_height: int = 200  # Height of the confidence plot in pixels
    plot_width: int = 800   # Width of the confidence plot


class Preprocessing:
    """Handles data transformations for the PyTorch model."""
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


def load_tetrahedron_from_obj(filepath: str) -> np.ndarray:
    """
    Load a tetrahedron from an OBJ file.
    Expects exactly 4 vertices forming the tetrahedron.
    Returns vertices as (4, 3) array.
    """
    vertices = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    vertices = np.array(vertices)
    if len(vertices) != 4:
        raise ValueError(f"Expected 4 vertices for a tetrahedron, got {len(vertices)}")
    
    return vertices


def create_default_tetrahedron() -> np.ndarray:
    """Create a default regular tetrahedron centered at origin."""
    # Regular tetrahedron vertices
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0]
    ], dtype=np.float64)


# =============================================================================
# Tetrahedron Utilities
# =============================================================================


class ConfidenceVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tetrahedra data
        self.tet_stationary: Optional[np.ndarray] = None  # (4, 3)
        self.tet_moving_base: Optional[np.ndarray] = None  # (4, 3) - base position
        
        # Trajectory data
        self.x_positions: np.ndarray = np.array([])
        self.current_step: int = 0
        
        # Confidence history (append-only - records every computation)
        self.confidence_history: List[float] = []
        
    def load_model(self, model_path: str) -> bool:
        """Load the PyTorch collision detection model."""
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
        """Load tetrahedra from OBJ files or use defaults."""
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
            
            # Center tetrahedra at origin
            self.tet_stationary -= self.tet_stationary.mean(axis=0)
            self.tet_moving_base -= self.tet_moving_base.mean(axis=0)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load tetrahedra: {e}")
            return False
    
    def normalize_to_unit_cube(self, tet1: np.ndarray, tet2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize both tetrahedra so all vertices fit within the unit cube [0, 1]^3."""
        # Combine both tetrahedra to find global bounds
        combined = np.concatenate([tet1, tet2], axis=0)  # (8, 3)
        
        # Find min and max for each axis
        min_vals = combined.min(axis=0)
        max_vals = combined.max(axis=0)
        
        # Compute the range for each axis
        range_vals = max_vals - min_vals
        
        # Use the maximum range to maintain aspect ratio (uniform scaling)
        max_range = range_vals.max()
        
        # Avoid division by zero
        if max_range == 0:
            max_range = 1.0
        
        # Normalize: shift to origin, then scale to [0, 1]
        tet1_normalized = (tet1 - min_vals) / max_range
        tet2_normalized = (tet2 - min_vals) / max_range
        
        return tet1_normalized, tet2_normalized
    
    def compute_confidence(self, tet1: np.ndarray, tet2: np.ndarray) -> float:
        """Compute the collision confidence for a pair of tetrahedra."""
        if self.model is None:
            return 0.0
        
        # Normalize both tetrahedra to fit within unit cube
        tet1_norm, tet2_norm = self.normalize_to_unit_cube(tet1, tet2)
        
        # Flatten and concatenate: 24 values (4*3 + 4*3)
        data = np.concatenate([tet1_norm.flatten(), tet2_norm.flatten()])
        tensor = torch.tensor(data, dtype=torch.float64).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            transformed = Preprocessing.principal_axis_transform(tensor)
            output = self.model(transformed)
            # Apply sigmoid to get probability in [0, 1]
            confidence = torch.sigmoid(output).cpu().numpy().flatten()[0]
        
        return float(confidence)
    
    def initialize_steps(self):
        """Initialize the trajectory step positions and reset history."""
        x_min, x_max = self.config.x_range
        self.x_positions = np.linspace(x_min, x_max, self.config.num_positions)
        self.confidence_history = []
        self.current_step = 0
        logger.info(f"Initialized {self.config.num_positions} steps from x={x_min:.2f} to x={x_max:.2f}")
    
    def _get_moved_tetrahedron(self, step: int) -> np.ndarray:
        """Get the moving tetrahedron at a specific step position."""
        x_offset = self.x_positions[step]
        tet_moved = self.tet_moving_base.copy()
        tet_moved[:, 0] += x_offset
        return tet_moved
    
    def get_current_confidence(self) -> float:
        """Compute and return confidence for current step position."""
        tet_moved = self._get_moved_tetrahedron(self.current_step)
        return self.compute_confidence(self.tet_stationary, tet_moved)
    
    def update_history_for_step(self, new_step: int):
        """Update step and append confidence to history (append-only, never deletes)."""
        if new_step == self.current_step:
            return
        
        self.current_step = new_step
        tet_moved = self._get_moved_tetrahedron(new_step)
        conf = self.compute_confidence(self.tet_stationary, tet_moved)
        self.confidence_history.append(conf)
    
    def precompute_all_confidences(self):
        """Precompute all confidence values along the trajectory (used for export)."""
        logger.info("Precomputing all confidence values...")
        
        self.initialize_steps()
        
        for step in range(self.config.num_positions):
            tet_moved = self._get_moved_tetrahedron(step)
            conf = self.compute_confidence(self.tet_stationary, tet_moved)
            self.confidence_history.append(conf)
        
        logger.info(f"Precomputed {len(self.confidence_history)} confidence values")
        logger.info(f"Confidence range: [{min(self.confidence_history):.3f}, {max(self.confidence_history):.3f}]")
    
    # =========================================================================
    # Polyscope Visualization
    # =========================================================================
    
    def initialize_polyscope(self):
        """Initialize Polyscope visualization with both tetrahedra."""
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        
        # Register stationary tetrahedron (wireframe)
        ps_stationary = ps.register_curve_network(
            "Stationary Tetrahedron",
            self.tet_stationary,
            TETRAHEDRON_EDGES
        )
        ps_stationary.set_color(COLOR_STATIONARY)
        ps_stationary.set_radius(WIREFRAME_RADIUS)
        
        # Register moving tetrahedron
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron",
            self._get_moved_tetrahedron(self.current_step),
            TETRAHEDRON_EDGES
        )
        ps_moving.set_color(COLOR_NO_COLLISION)
        ps_moving.set_radius(WIREFRAME_RADIUS)
        
        logger.info("Polyscope visualization initialized")
    
    def update_visualization(self):
        """Update the moving tetrahedron position and color based on collision state."""
        current_conf = self.get_current_confidence()
        is_collision = current_conf > COLLISION_THRESHOLD
        
        # Re-register to update position (Polyscope requires re-registration)
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron",
            self._get_moved_tetrahedron(self.current_step),
            TETRAHEDRON_EDGES
        )
        ps_moving.set_color(COLOR_COLLISION if is_collision else COLOR_NO_COLLISION)
        ps_moving.set_radius(WIREFRAME_RADIUS)
    
    # =========================================================================
    # UI Components
    # =========================================================================
    
    def _render_status_panel(self, confidence: float, x_position: float):
        """Render the status information panel."""
        is_collision = confidence > COLLISION_THRESHOLD
        
        # Colors for collision state
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
        
        # Collision status indicator
        if is_collision:
            psim.TextColored((1.0, 0.35, 0.35, 1.0), ">> INTERSECTION DETECTED <<")
        else:
            psim.TextColored((0.35, 0.9, 0.35, 1.0), "No Intersection")
    
    def _render_confidence_graph(self):
        """Render the confidence history graph."""
        psim.TextColored((0.8, 0.8, 0.8, 1.0), "Confidence Curve")
        
        if self.confidence_history:
            psim.PlotLines(
                "##confidence_plot",
                self.confidence_history.copy(),
                graph_size=(400, 140),
                scale_min=0.0,
                scale_max=1.0
            )
            psim.TextColored((0.6, 0.6, 0.6, 1.0), f"Samples: {len(self.confidence_history)}")
        else:
            psim.TextColored((0.5, 0.5, 0.5, 1.0), "Move slider to build curve...")
        
        psim.TextColored((0.5, 0.5, 0.5, 1.0), f"Threshold: {COLLISION_THRESHOLD}")
    
    def _render_confidence_bar(self, confidence: float):
        """Render the confidence progress bar."""
        bar_width = 50
        filled = int(confidence * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        color = (1.0, 0.4, 0.4, 1.0) if confidence > COLLISION_THRESHOLD else (0.4, 0.9, 0.4, 1.0)
        psim.TextColored(color, f"{bar} {confidence:.1%}")
    
    def create_ui_callback(self):
        """Create the Polyscope UI callback with slider-controlled visualization."""
        
        def ui_callback():
            current_conf = self.get_current_confidence()
            current_x = self.x_positions[self.current_step]
            
            # Header
            psim.TextColored((0.9, 0.9, 0.9, 1.0), "COLLISION DETECTION")
            psim.Separator()
            
            # Status panel
            self._render_status_panel(current_conf, current_x)
            psim.Separator()
            
            # Step slider control
            changed, new_step = psim.SliderInt(
                "##step_slider", 
                self.current_step, 
                0, 
                self.config.num_positions - 1
            )
            if changed and new_step != self.current_step:
                self.update_history_for_step(new_step)
                self.update_visualization()
            
            # Reset button
            psim.SameLine()
            if psim.Button("Reset"):
                self.initialize_steps()
                self.update_visualization()
            
            psim.Separator()
            
            # Confidence graph
            self._render_confidence_graph()
            
            # Confidence bar
            psim.Separator()
            self._render_confidence_bar(current_conf)
        
        return ui_callback
    
    def run(self):
        """Run the visualization."""
        if self.tet_stationary is None or self.tet_moving_base is None:
            logger.error("Tetrahedra not loaded")
            return
        
        if self.model is None:
            logger.warning("No model loaded - confidence values will be 0")
        
        # Initialize steps
        self.initialize_steps()
        
        # Compute initial position (step 0)
        self.get_current_confidence()
        
        # Initialize and run Polyscope
        self.initialize_polyscope()
        ps.set_user_callback(self.create_ui_callback())
        ps.show()
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_confidence_plot(self, output_path: str = "confidence_curve.png", 
                                figsize: Tuple[float, float] = (12, 4)):
        """Export the confidence curve as a publication-quality matplotlib plot."""
        self.precompute_all_confidences()
        
        fig, ax = plt.subplots(figsize=figsize)
        conf_array = np.array(self.confidence_history)
        
        # Plot curve segments colored by collision state
        for i in range(len(self.x_positions) - 1):
            x = [self.x_positions[i], self.x_positions[i + 1]]
            y = [conf_array[i], conf_array[i + 1]]
            color = 'red' if conf_array[i] > COLLISION_THRESHOLD else 'green'
            ax.plot(x, y, color=color, linewidth=2)
        
        # Fill regions based on threshold
        above_threshold = conf_array > COLLISION_THRESHOLD
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=above_threshold, alpha=0.3, color='red', 
                        label='Collision detected')
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=~above_threshold, alpha=0.3, color='green',
                        label='No collision')
        
        # Threshold line
        ax.axhline(y=COLLISION_THRESHOLD, color='gray', linestyle='--', linewidth=1.5, 
                   label=f'Decision threshold ({COLLISION_THRESHOLD})')
        
        # Labels and styling
        ax.set_xlabel('X Position (tetrahedron offset)', fontsize=12)
        ax.set_ylabel('Model Confidence', fontsize=12)
        ax.set_title('Neural Network Collision Detection Confidence', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(self.config.x_range[0], self.config.x_range[1])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key regions
        max_conf_idx = np.argmax(conf_array)
        max_conf_x = self.x_positions[max_conf_idx]
        max_conf_y = conf_array[max_conf_idx]
        ax.annotate(f'Peak: {max_conf_y:.2f}', 
                    xy=(max_conf_x, max_conf_y), 
                    xytext=(max_conf_x + 0.5, max_conf_y + 0.1),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confidence plot saved to {output_path}")
        plt.close()
        
        return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize collision detection confidence for tetrahedra"
    )
    parser.add_argument(
        "--stationary", "-s",
        type=str,
        default=None,
        help="Path to stationary tetrahedron OBJ file (default: use built-in)"
    )
    parser.add_argument(
        "--moving", "-m",
        type=str,
        default=None,
        help="Path to moving tetrahedron OBJ file (default: use built-in)"
    )
    parser.add_argument(
        "--model", "-M",
        type=str,
        default="model/EuroGraphicsPaper.pt",
        help="Path to PyTorch model file"
    )
    parser.add_argument(
        "--positions", "-n",
        type=int,
        default=100,
        help="Number of positions along trajectory (default: 100)"
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=-3.0,
        help="Minimum X offset (default: -3.0)"
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=3.0,
        help="Maximum X offset (default: 3.0)"
    )
    parser.add_argument(
        "--export", "-e",
        type=str,
        default=None,
        help="Export confidence curve to file (e.g., confidence_curve.png). If provided, skips interactive visualization."
    )
    parser.add_argument(
        "--export-and-show",
        action="store_true",
        help="Export confidence curve and also show interactive visualization"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = VisualizationConfig(
        num_positions=args.positions,
        x_range=(args.x_min, args.x_max)
    )
    
    # Create visualizer
    visualizer = ConfidenceVisualizer(config)
    
    # Load model
    visualizer.load_model(args.model)
    
    # Load tetrahedra
    if not visualizer.load_tetrahedra(args.stationary, args.moving):
        logger.error("Failed to load tetrahedra")
        return
    
    # Handle export options
    if args.export:
        visualizer.export_confidence_plot(args.export)
        if not args.export_and_show:
            return
    elif args.export_and_show:
        visualizer.export_confidence_plot("confidence_curve.png")
    
    # Run interactive visualization
    visualizer.run()


if __name__ == "__main__":
    main()
