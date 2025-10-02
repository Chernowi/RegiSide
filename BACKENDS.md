# MAPPO Training Framework - Backend Selection

## Default: PyTorch + CUDA

By default, the training framework uses **PyTorch with CUDA** for GPU-accelerated training. This provides:
- Easy installation and setup
- Excellent CUDA support for NVIDIA GPUs
- Familiar PyTorch API
- Good debugging experience

## Optional: JAX Backend

You can optionally use JAX for training by:

1. Installing JAX dependencies:
```bash
pip install jax jaxlib flax optax chex
```

2. Setting `backend: "jax"` in your config file:
```yaml
algorithm: "mappo"
backend: "jax"  # Enable JAX backend
...
```

### When to use JAX?
- Need XLA compilation for maximum performance
- Working with TPUs
- Want functional programming approach
- Require automatic vectorization (vmap)

### When to use PyTorch (default)?
- Using NVIDIA GPUs with CUDA
- Want familiar PyTorch ecosystem
- Need easier debugging
- Prefer standard installation

## Configuration

Edit `configs/mappo_default.yaml`:

```yaml
# Backend selection (default: pytorch)
backend: "pytorch"  # Options: pytorch, jax

# Hardware (automatically uses CUDA if available)
device: "auto"  # Options: auto, cpu, cuda, gpu
```

## Quick Start

```bash
# PyTorch (default) - uses CUDA automatically if available
python train.py --config configs/mappo_default.yaml

# Force CPU
python train.py --config configs/mappo_default.yaml --device cpu

# JAX backend (requires JAX installed)
# First, edit configs/mappo_default.yaml and set: backend: "jax"
python train.py --config configs/mappo_default.yaml
```

See `docs/TRAINING.md` for full documentation.
