"""PyTorch 35-neuron validation: an independent reimplementation.

Runs the core experiment (Baseline, A, C1, C2, D') under static and VN and
writes its own results CSV, reproducing the fake-mirror effect in PyTorch. If the
public NumPy repo (fake-mirror-effect) is cloned as a sibling directory, it also
prints a PyTorch-vs-NumPy cross-check; otherwise that comparison is skipped and
the PyTorch run is standalone.

Usage:
    python run_validation.py                    # He normal (matches NumPy)
    python run_validation.py --init kaiming     # PyTorch default Kaiming uniform
    python run_validation.py --init he_normal   # He normal (explicit)
"""

import os
import sys
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

import argparse

from network import RecurrentMLP, ParamMatchedFF

INIT_SCHEME = 'he_normal'  # set by argparse


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data(n_samples, noise_level=0.5, seed=0):
    """Generate 5-class pattern classification data (matches NumPy version)."""
    rng = np.random.RandomState(seed)
    prototypes = np.zeros((5, 10))
    for k in range(5):
        prototypes[k, 2*k] = 1.0
        prototypes[k, 2*k+1] = 1.0
        # Modulo wrap-around: matches NumPy src/training.py exactly
        prototypes[k, (2*k + 2) % 10] = 0.3
        prototypes[k, (2*k - 1) % 10] = 0.3

    labels = rng.randint(0, 5, n_samples)
    X = prototypes[labels] + rng.randn(n_samples, 10) * noise_level
    y = np.zeros((n_samples, 5))
    y[np.arange(n_samples), labels] = 1.0
    return X.astype(np.float32), y.astype(np.float32), labels


def generate_data_vn(n_samples, noise_level=0.5, T=3, seed=0):
    """Generate variable-noise sequences."""
    rng = np.random.RandomState(seed)
    prototypes = np.zeros((5, 10))
    for k in range(5):
        prototypes[k, 2*k] = 1.0
        prototypes[k, 2*k+1] = 1.0
        prototypes[k, (2*k + 2) % 10] = 0.3
        prototypes[k, (2*k - 1) % 10] = 0.3

    labels = rng.randint(0, 5, n_samples)
    X_seq = np.zeros((n_samples, T, 10), dtype=np.float32)
    for t in range(T):
        X_seq[:, t, :] = prototypes[labels] + rng.randn(n_samples, 10) * noise_level
    y = np.zeros((n_samples, 5), dtype=np.float32)
    y[np.arange(n_samples), labels] = 1.0
    return X_seq, y, labels


def train_model(net, X, y, epochs=1000, lr=0.01, T=3, time_weights=(0.0, 0.2, 1.0),
                vn=False):
    """Train with 3-step BPTT and weighted-average time-weighted loss.

    epochs=1000 matches the standardized protocol of the main repo (joint
    static/VN convergence point); the loss is the weighted average
    (sum(w_t * CE_t) / sum(w)), so the effective learning rate is lr, not
    lr * sum(w).
    """
    # vn is accepted for call-site symmetry only; training is identical for static/VN (the data shape differs).
    optimizer = SGD(net.parameters(), lr=lr)
    tw = torch.tensor(time_weights, dtype=torch.float32)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    labels = torch.argmax(y_t, dim=1)

    net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X_t, T=T)

        loss = torch.tensor(0.0)
        for t in range(T):
            if tw[t] > 0:
                loss = loss + tw[t] * F.cross_entropy(outputs[t], labels)
        loss = loss / tw.sum()

        loss.backward()
        optimizer.step()


def evaluate(net, X, y_onehot, labels, T=3, feedback_mode='self',
             clone=None, scramble_seed=None):
    """Evaluate accuracy at t=1 and t=3."""
    net.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X)
        scramble_rng = None
        if feedback_mode == 'scrambled' and scramble_seed is not None:
            scramble_rng = torch.Generator()
            scramble_rng.manual_seed(scramble_seed)

        outputs = net(X_t, T=T, feedback_mode=feedback_mode,
                      clone=clone, scramble_rng=scramble_rng)

        pred_t1 = outputs[0].argmax(dim=1).numpy()
        pred_t3 = outputs[T-1].argmax(dim=1).numpy()

        acc_t1 = (pred_t1 == labels).mean()
        acc_t3 = (pred_t3 == labels).mean()
        return float(acc_t1), float(acc_t3), float(acc_t3 - acc_t1)


def run_one_seed(seed, noise=0.5, T=3, epochs=1000, setting='static'):
    """Run all conditions for one seed."""
    set_seed(seed)

    # Generate data
    if setting == 'static':
        X_train, y_train, labels_train = generate_data(200, noise, seed=seed)
        X_test, y_test, labels_test = generate_data(200, noise, seed=seed + 500)
    else:
        X_train, y_train, labels_train = generate_data_vn(200, noise, T=T, seed=seed)
        X_test, y_test, labels_test = generate_data_vn(200, noise, T=T, seed=seed + 500)

    # Train target
    set_seed(seed)
    target = RecurrentMLP(feedback_tau=2.0, init=INIT_SCHEME)
    train_model(target, X_train, y_train, epochs=epochs, T=T, vn=(setting == 'vn'))

    # Baseline
    acc_t1, acc_t3, gain = evaluate(target, X_test, y_test, labels_test, T=T)
    results = {'seed': seed, 'setting': setting}
    results['bl_acc_t1'] = acc_t1
    results['bl_acc_t3'] = acc_t3
    results['bl_gain'] = gain

    # Group A (ablated)
    acc_t1, acc_t3, gain = evaluate(target, X_test, y_test, labels_test, T=T,
                                     feedback_mode='ablated')
    results['a_gain'] = gain

    # Group C1 (scrambled)
    acc_t1, acc_t3, gain = evaluate(target, X_test, y_test, labels_test, T=T,
                                     feedback_mode='scrambled', scramble_seed=seed * 1000)
    results['c1_gain'] = gain

    # Train clone for C2
    donor_seed = seed + 100
    set_seed(donor_seed)
    clone = RecurrentMLP(feedback_tau=2.0, init=INIT_SCHEME)
    if setting == 'static':
        X_clone, y_clone, _ = generate_data(200, noise, seed=donor_seed)
    else:
        X_clone, y_clone, _ = generate_data_vn(200, noise, T=T, seed=donor_seed)
    train_model(clone, X_clone, y_clone, epochs=epochs, T=T, vn=(setting == 'vn'))

    # Group C2 (clone feedback)
    acc_t1, acc_t3, gain = evaluate(target, X_test, y_test, labels_test, T=T,
                                     feedback_mode='clone', clone=clone)
    results['c2_gain'] = gain

    # D' (parameter-matched FF)
    set_seed(seed)
    dp = ParamMatchedFF()
    if setting == 'static':
        train_dp(dp, X_train, y_train, epochs=epochs)
    else:
        train_dp(dp, X_train[:, -1, :], y_train, epochs=epochs)
    dp.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test if setting == 'static' else X_test[:, -1, :])
        pred = dp(X_t).argmax(dim=1).numpy()
        dp_acc = (pred == labels_test).mean()
    results['dp_acc'] = float(dp_acc)
    results['dp_gain'] = 0.0  # single-pass, no temporal

    return results


def train_dp(net, X, y, epochs=1000, lr=0.01):
    """Train feedforward model."""
    optimizer = SGD(net.parameters(), lr=lr)
    X_t = torch.from_numpy(X)
    labels = torch.argmax(torch.from_numpy(y), dim=1)
    net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = net(X_t)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()


def main():
    global INIT_SCHEME
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', choices=['he_normal', 'kaiming', 'xavier'],
                        default='he_normal', help='Weight initialization scheme')
    args = parser.parse_args()
    INIT_SCHEME = args.init

    N_SEEDS = 20
    results_all = []

    print("=" * 60)
    print(f"PyTorch 35-Neuron Cross-Validation (init={INIT_SCHEME})")
    print("=" * 60)
    t0 = time.time()

    for setting in ['static', 'vn']:
        print(f"\n--- {setting.upper()} ---")
        for seed in range(N_SEEDS):
            r = run_one_seed(seed, setting=setting)
            results_all.append(r)
            print(f"  seed={seed}: BL={r['bl_gain']:+.3f} A={r['a_gain']:+.3f} "
                  f"C1={r['c1_gain']:+.3f} C2={r['c2_gain']:+.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save CSV
    os.makedirs('results', exist_ok=True)
    fields = ['seed', 'setting', 'bl_acc_t1', 'bl_acc_t3', 'bl_gain',
              'a_gain', 'c1_gain', 'c2_gain', 'dp_acc', 'dp_gain']
    csv_name = f'results/pytorch_validation_{INIT_SCHEME}.csv'
    with open(csv_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results_all)

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- SD)")
    print("=" * 60)

    for setting in ['static', 'vn']:
        rows = [r for r in results_all if r['setting'] == setting]
        print(f"\n  [{setting.upper()}]")
        for key, label in [('bl_gain', 'Baseline'), ('a_gain', 'Group A'),
                           ('c1_gain', 'C1'), ('c2_gain', 'C2')]:
            vals = [r[key] for r in rows]
            print(f"    {label:12s}: {np.mean(vals):+.4f} +/- {np.std(vals):.4f}")

    # Optional cross-check against the NumPy reference results: clone the public
    # fake-mirror-effect repo as a sibling directory to enable it. Skipped
    # otherwise (the PyTorch run above is standalone and self-contained).
    numpy_csv = os.path.join('..', 'fake-mirror-effect', 'results', 'raw_metrics.csv')
    if os.path.exists(numpy_csv):
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION vs NumPy")
        print("=" * 60)

        numpy_data = {}
        with open(numpy_csv) as f:
            for row in csv.DictReader(f):
                if float(row['noise_level']) == 0.5:
                    grp = row['group']
                    if grp not in numpy_data:
                        numpy_data[grp] = []
                    numpy_data[grp].append(float(row['gain']))

        # Average B1 repeats
        if 'B1' in numpy_data:
            from collections import defaultdict
            b1_per_seed = defaultdict(list)
            with open(numpy_csv) as f:
                for row in csv.DictReader(f):
                    if float(row['noise_level']) == 0.5 and row['group'] == 'B1':
                        b1_per_seed[int(row['seed_model'])].append(float(row['gain']))
            numpy_data['B1'] = [np.mean(v) for v in b1_per_seed.values()]

        # For groups with repeats (B1=30, C1=30), average per seed first
        for grp in ['B1', 'C1']:
            if grp in numpy_data and len(numpy_data[grp]) > 20:
                per_seed = defaultdict(list)
                with open(numpy_csv) as f:
                    for row in csv.DictReader(f):
                        if float(row['noise_level']) == 0.5 and row['group'] == grp:
                            per_seed[int(row['seed_model'])].append(float(row['gain']))
                numpy_data[grp] = [np.mean(v) for v in per_seed.values()]

        pt_static = [r for r in results_all if r['setting'] == 'static']
        group_map = {'Baseline': 'bl_gain', 'A': 'a_gain', 'C1': 'c1_gain', 'C2': 'c2_gain'}

        print("\n  Static (noise=0.5):")
        print(f"  {'Group':12s} {'NumPy':>10s} {'PyTorch':>10s} {'Diff':>10s}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        for grp, key in group_map.items():
            if grp in numpy_data:
                np_mean = np.mean(numpy_data[grp])
                pt_mean = np.mean([r[key] for r in pt_static])
                diff = pt_mean - np_mean
                print(f"  {grp:12s} {np_mean:+10.4f} {pt_mean:+10.4f} {diff:+10.4f}")

    print(f"\nSaved: {csv_name}")
    print(f"Init scheme: {INIT_SCHEME}")
    print("Done!")


if __name__ == '__main__':
    main()
