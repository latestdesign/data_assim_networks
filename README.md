# Data Assimilation Networks (DAN)

Implementation of Data Assimilation Networks for state estimation in observed dynamical systems.

## Experiments

- **Linear 2D** — Periodic linear dynamics, fixed horizon training with L-BFGS + BPTT
- **Lorenz 96 (40D)** — Chaotic dynamics, online training with Adam + truncated BPTT

## Usage

```bash
# Run from notebook
jupyter notebook tp.ipynb

# Or run from command line
python main.py -save lin2d_exp.py [-run | -plot]
python main.py -save lorenz_exp.py [-run | -plot]
```

## Pre-trained weights

Set `RETRAIN = False` in the notebook to load pre-trained checkpoints.
If checkpoints are not present locally, set `RETRAIN = True` to train from scratch.

## Environment
```bash
conda env create -f environment.yml -p .venv
conda activate ./.venv
```

Or use your own Python environment with PyTorch installed.

## Files

| File | Description |
|------|-------------|
| `tp.ipynb` | Main notebook with both experiments |
| `filters.py` | DAN modules, Gaussian distributions, dynamics |
| `manage_exp.py` | Training loops (full, online), testing, checkpointing |
| `utils.py` | Simulation and plotting utilities |
| `lin2d_exp.py` | Linear 2D experiment configuration |
| `lorenz_exp.py` | Lorenz 96 experiment configuration |
| `main.py` | Command-line experiment runner |
