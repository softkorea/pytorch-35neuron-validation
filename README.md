# PyTorch 35-Neuron Cross-Validation

Independent PyTorch reimplementation of the 35-neuron RecurrentMLP to verify that the "Fake Mirror Effect" is not an artifact of the NumPy implementation.

## Protocol

Models are trained for 1000 epochs of 3-step BPTT with a weighted-average time-weighted loss (step weights `(0, 0.2, 1.0)`, normalized by their sum so the effective learning rate equals `lr`), full-batch SGD at `lr=0.01`. This matches the standardized training protocol of the main repository (joint static/VN convergence point).

## Usage

```bash
pip install torch numpy

# He normal initialization (matches the NumPy codebase)
python run_validation.py --init he_normal

# Kaiming uniform (PyTorch default — self-correction does NOT emerge)
python run_validation.py --init kaiming
```

The script runs standalone and writes its own results. To additionally print a
PyTorch-vs-NumPy cross-check, clone the public NumPy repo as a sibling directory:

```bash
git clone https://github.com/softkorea/fake-mirror-effect ../fake-mirror-effect
```

## Key Results (N=20, mean gain)

| Init | Setting | Baseline | Group A | C1 | C2 |
|------|---------|----------|---------|------|------|
| He normal | Static | +0.059 | +0.000 | -0.095 | -0.048 |
| He normal | VN | +0.156 | +0.004 | -0.129 | -0.005 |
| Kaiming uniform | Static | +0.002 | +0.000 | -0.001 | +0.001 |
| Kaiming uniform | VN | -0.000 | -0.001 | -0.003 | -0.003 |

Under He normal initialization the qualitative pattern replicates: a positive Baseline self-correction gain, a near-zero recurrence-off control (Group A), and negative shuffled-feedback (C1) and clone-feedback (C2) gains. Kaiming uniform (~6x smaller initial variance) does not produce self-correction, consistent with initialization-dependent emergence.
