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
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def get_tetrahedron_edges() -> List[Tuple[int, int]]:
    """Return the 6 edges of a tetrahedron as vertex index pairs."""
    return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


class ConfidenceVisualizer:
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Tetrahedra data
        self.tet_stationary: Optional[np.ndarray] = None  # (4, 3)
        self.tet_moving_base: Optional[np.ndarray] = None  # (4, 3) - base position
        
        # Step data
        self.x_positions: np.ndarray = np.array([])
        self.confidence_history: List[float] = []  # Always grows - records every computation
        
        # Current state
        self.current_step: int = 0
        self.last_recorded_step: int = -1  # Track last step we recorded to avoid duplicates
        
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
        """Initialize the step positions."""
        x_min, x_max = self.config.x_range
        self.x_positions = np.linspace(x_min, x_max, self.config.num_positions)
        self.confidence_history = []
        self.current_step = 0
        self.last_recorded_step = -1
        logger.info(f"Initialized {self.config.num_positions} steps from x={x_min} to x={x_max}")
    
    def get_current_confidence(self) -> float:
        """Get confidence for current step position."""
        x_offset = self.x_positions[self.current_step]
        tet_moved = self.tet_moving_base.copy()
        tet_moved[:, 0] += x_offset
        return self.compute_confidence(self.tet_stationary, tet_moved)
    
    def update_history_for_step(self, new_step: int):
        """Update confidence history when step changes - append only, never delete."""
        old_step = self.current_step
        self.current_step = new_step
        
        # Only append if step actually changed
        if new_step != old_step:
            # Always compute and append the new confidence value
            x_offset = self.x_positions[new_step]
            tet_moved = self.tet_moving_base.copy()
            tet_moved[:, 0] += x_offset
            conf = self.compute_confidence(self.tet_stationary, tet_moved)
            self.confidence_history.append(conf)
    
    def precompute_all_confidences(self):
        """Precompute all confidence values (used for export)."""
        logger.info("Precomputing all confidence values...")
        
        x_min, x_max = self.config.x_range
        self.x_positions = np.linspace(x_min, x_max, self.config.num_positions)
        self.confidence_history = []
        
        for i, x_offset in enumerate(self.x_positions):
            tet_moved = self.tet_moving_base.copy()
            tet_moved[:, 0] += x_offset
            conf = self.compute_confidence(self.tet_stationary, tet_moved)
            self.confidence_history.append(conf)
        
        logger.info(f"Precomputed {len(self.confidence_history)} confidence values")
        logger.info(f"Confidence range: [{min(self.confidence_history):.3f}, {max(self.confidence_history):.3f}]")
    
    def get_current_moving_tet(self) -> np.ndarray:
        """Get the current position of the moving tetrahedron."""
        x_offset = self.x_positions[self.current_step]
        tet_moved = self.tet_moving_base.copy()
        tet_moved[:, 0] += x_offset
        return tet_moved
    
    def initialize_polyscope(self):
        """Initialize Polyscope visualization."""
        ps.init()
        ps.set_up_dir("y_up")
        ps.set_ground_plane_mode("none")
        
        # Register stationary tetrahedron as curve network (wireframe)
        edges = get_tetrahedron_edges()
        edge_array = np.array(edges)
        
        ps_stationary = ps.register_curve_network(
            "Stationary Tetrahedron",
            self.tet_stationary,
            edge_array
        )
        ps_stationary.set_color([0.2, 0.5, 0.9])  # Blue
        ps_stationary.set_radius(0.02)
        
        # Register moving tetrahedron
        initial_moving = self.get_current_moving_tet()
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron",
            initial_moving,
            edge_array
        )
        ps_moving.set_color([0.9, 0.5, 0.2])  # Orange
        ps_moving.set_radius(0.02)
        
        logger.info("Polyscope visualization initialized")
    
    def update_visualization(self):
        """Update the moving tetrahedron position in Polyscope."""
        moving_tet = self.get_current_moving_tet()
        edges = np.array(get_tetrahedron_edges())
        
        # Get current confidence
        current_conf = self.get_current_confidence()
        is_collision = current_conf > 0.5
        
        # Re-register to update position
        ps_moving = ps.register_curve_network(
            "Moving Tetrahedron",
            moving_tet,
            edges
        )
        # Red if collision detected, green otherwise
        if is_collision:
            ps_moving.set_color([1.0, 0.2, 0.2])  # Red - collision
        else:
            ps_moving.set_color([0.2, 0.8, 0.2])  # Green - no collision
        ps_moving.set_radius(0.02)
    
    def create_ui_callback(self):
        """Create the Polyscope UI callback with slider-controlled confidence graph."""
        
        def ui_callback():
            # Get current confidence
            current_conf = self.get_current_confidence()
            current_x = self.x_positions[self.current_step]
            
            # Title
            psim.TextColored((0.9, 0.9, 0.9, 1.0), "COLLISION DETECTION")
            psim.Separator()
            
            # Status display
            psim.Text(f"Step")
            psim.SameLine()
            psim.TextColored((0.7, 0.85, 1.0, 1.0), f"{self.current_step + 1} / {self.config.num_positions}")
            
            psim.Text(f"Position")
            psim.SameLine()
            psim.TextColored((0.7, 0.85, 1.0, 1.0), f"x = {current_x:+.3f}")
            
            psim.Text(f"Confidence")
            psim.SameLine()
            if current_conf > 0.5:
                psim.TextColored((1.0, 0.4, 0.4, 1.0), f"{current_conf:.4f}")
            else:
                psim.TextColored((0.4, 1.0, 0.4, 1.0), f"{current_conf:.4f}")
            
            psim.Separator()
            
            # Status indicator - clean and prominent
            if current_conf > 0.5:
                psim.TextColored((1.0, 0.35, 0.35, 1.0), ">> INTERSECTION DETECTED <<")
            else:
                psim.TextColored((0.35, 0.9, 0.35, 1.0), "No Intersection")
            
            psim.Separator()
            
            # Step slider - the main control
            changed, new_step = psim.SliderInt(
                "##step_slider", 
                self.current_step, 
                0, 
                self.config.num_positions - 1
            )
            if changed and new_step != self.current_step:
                self.update_history_for_step(new_step)
                self.update_visualization()
            
            psim.SameLine()
            if psim.Button("Reset"):
                self.current_step = 0
                self.confidence_history = []
                self.last_recorded_step = -1
                self.update_visualization()
            
            psim.Separator()
            
            # Confidence graph section
            psim.TextColored((0.8, 0.8, 0.8, 1.0), "Confidence Curve")
            
            if len(self.confidence_history) > 0:
                graph_values = self.confidence_history.copy()
                
                # Plot the confidence curve - larger and cleaner
                psim.PlotLines(
                    "##confidence_plot",
                    graph_values,
                    graph_size=(400, 140),
                    scale_min=0.0,
                    scale_max=1.0
                )
                
                # Graph info
                psim.TextColored((0.6, 0.6, 0.6, 1.0), f"Samples: {len(self.confidence_history)}")
            else:
                psim.TextColored((0.5, 0.5, 0.5, 1.0), "Move slider to build curve...")
            
            # Threshold indicator
            psim.TextColored((0.5, 0.5, 0.5, 1.0), "Threshold: 0.5")
            
            # Confidence bar - cleaner visualization
            psim.Separator()
            bar_width = 50
            filled = int(current_conf * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            if current_conf > 0.5:
                psim.TextColored((1.0, 0.4, 0.4, 1.0), f"{bar} {current_conf:.1%}")
            else:
                psim.TextColored((0.4, 0.9, 0.4, 1.0), f"{bar} {current_conf:.1%}")
        
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
    
    def export_confidence_plot(self, output_path: str = "confidence_curve.png", 
                                figsize: Tuple[float, float] = (12, 4)):
        """Export the confidence curve as a publication-quality matplotlib plot."""
        # Precompute all confidences for export
        self.precompute_all_confidences()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert to numpy array for plotting
        conf_array = np.array(self.confidence_history)
        
        # Plot confidence curve with color based on threshold
        for i in range(len(self.x_positions) - 1):
            x = [self.x_positions[i], self.x_positions[i + 1]]
            y = [conf_array[i], conf_array[i + 1]]
            color = 'red' if conf_array[i] > 0.5 else 'green'
            ax.plot(x, y, color=color, linewidth=2)
        
        # Fill regions
        above_threshold = conf_array > 0.5
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=above_threshold, alpha=0.3, color='red', 
                        label='Collision detected')
        ax.fill_between(self.x_positions, 0, conf_array, 
                        where=~above_threshold, alpha=0.3, color='green',
                        label='No collision')
        
        # Threshold line
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, 
                   label='Decision threshold (0.5)')
        
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
