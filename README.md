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

This launches the tetrahedral collision visualization using the default model and sample mesh data.

```bash
python3 mesh_intersection.py
```

If the model file is missing, place it at:

- `model/TetrahedronPairNet_L_principal_axis.pt`

The script loads sample meshes from:

- `data/armadillo.obj`

## Run the confidence visualization

Use this script to visualize how the neural model predicts collision confidence as one tetrahedron moves through another.

```bash
python3 confidence_visualization.py
```

Optional arguments:

- `-M`, `--model` : path to a PyTorch model file
- `-s`, `--stationary` : stationary tetrahedron OBJ file
- `-m`, `--moving` : moving tetrahedron OBJ file
- `-e`, `--export` : export the confidence plot to a file
- `--export-and-show` : export and then show the interactive visualization

Example:

```bash
python3 confidence_visualization.py --model model/EuroGraphicsDemo_PCA_Unit.pt --export confidence.png
```

## Notes

- The repo expects `data/` and `model/` directories to exist.
- `tetgen` is required for mesh tetrahedralization.
- The simulator uses `polyscope` for interactive visualization.
