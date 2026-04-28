# Tetrahedral Collision Simulator

Simple instructions for installing and running the simulation.

## Install

1. Open a terminal in the repository root.
2. Create and activate a Python virtual environment.

On Linux/WSL/macOS:

If `python3 -m venv venv` fails because `ensurepip` is not available, update packages and install the venv support package first:

```bash
sudo apt update
sudo apt install python3-venv
```

Then create and activate the environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install Python dependencies inside the environment:

```bash
python -m pip install -r requirements.txt
```

> If your system uses `python` instead of `python3`, use that command consistently.

## Run the main collision simulator

This script runs the basic tetrahedral collision simulation and opens an interactive `polyscope` viewer.

```bash
python3 mesh_intersection.py
```

It uses the default model and sample mesh data:

- model: `model/TetrahedronPairNet_L_principal_axis.pt`
- geometry: `data/spot.obj`

In `mesh_intersection.py`, the simulation is configured with `SimulationConfig`. To make the static mesh rotate while the other mesh moves, set `rotate_static=True` in the configuration.

## Run the confidence visualization

This script loads a PyTorch collision model and evaluates one tetrahedron moving along a trajectory past another.

```bash
python3 confidence_visualization.py
```

It can also export a confidence curve plot without launching the interactive viewer.

Optional arguments:

- `-M`, `--model` : path to a PyTorch model file
- `-s`, `--stationary` : stationary tetrahedron OBJ file
- `-m`, `--moving` : moving tetrahedron OBJ file
- `-e`, `--export` : export the confidence plot to a file
- `--export-and-show` : export the plot and show the interactive visualization

Example:

```bash
python3 confidence_visualization.py --model model/EuroGraphicsDemo_PCA_Unit.pt --export confidence.png
```

## Notes

- The repo expects `data/` and `model/` directories to exist.
- `tetgen` is required for mesh tetrahedralization.
- The simulator uses `polyscope` for interactive visualization.
