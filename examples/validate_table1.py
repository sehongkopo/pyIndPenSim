"""Reproduce the batch-yield rows of Table 1: pyIndPenSim against the released IndPenSim batches.

The released dataset of Goldrick et al. (2019) contains 100 batches: 1-30 recipe-driven with
Raman-based control off, 31-60 operator-controlled, 61-90 under Raman-based PAA control and
91-100 with faults. Its per-batch summary, ``100_Batches_IndPenSim_Statistics.csv``, is
distributed with that dataset and is not included here; pass its path with ``--stats``.

The script simulates N batches (default 100, seeds 2000 to 2000 + N - 1) with Raman-based
control off (``Raman_spec = 0``) and on (``Raman_spec = 2``), both with operator-like PRBS
perturbations and uneven batch length, and compares the net yields with released batches
1-30 and 61-90 by the difference in means, Cohen's d, and the Kolmogorov-Smirnov and
Mann-Whitney U tests.

    pip install -e ".[validate]"
    python examples/validate_table1.py --stats /path/to/100_Batches_IndPenSim_Statistics.csv --n 100 --workers 8

Expected output with --n 100 (manuscript Table 1):
    Raman off   released 2,913 ± 768 (30)   pyIndPenSim 2,894 ± 802 (100)   -18 kg (-0.63 %)   d -0.023   KS p 0.856   MWU p 0.836
    Raman on    released 3,484 ± 308 (30)   pyIndPenSim 3,481 ± 487 (100)    -3 kg (-0.08 %)   d -0.006   KS p 0.278   MWU p 0.709
"""
import argparse, contextlib, io
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
from pyindpensim import indpensim_run, batch_flags, net_yield_kg


def simulate(job):
    seed, raman = job
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            X = indpensim_run(1, batch_flags(raman=raman), rng=np.random.default_rng(seed), rtol=1e-6, atol=1e-8, solv=2)
        return net_yield_kg(X)
    except Exception:
        return float("nan")


def released_yields(path):
    import pandas as pd
    s = pd.read_csv(path)
    y = s.iloc[:, 3].to_numpy(float) / 1000.0          # "Penicllin_yield_total (kg)" column is recorded in g
    return {0: y[0:30], 2: y[60:90]}


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="path to 100_Batches_IndPenSim_Statistics.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    ref = released_yields(a.stats)
    jobs = [(a.seed0 + i, r) for r in (0, 2) for i in range(a.n)]
    with ProcessPoolExecutor(a.workers) as ex:
        out = np.array(list(ex.map(simulate, jobs, chunksize=1)))
    for k, (raman, name) in enumerate(((0, "Raman off"), (2, "Raman on"))):
        sim = out[k * a.n:(k + 1) * a.n]; sim = sim[np.isfinite(sim)]; rel = ref[raman]
        diff = sim.mean() - rel.mean()
        print(f"{name:<10s}  released {rel.mean():,.0f} ± {rel.std(ddof=1):,.0f} ({len(rel)})   "
              f"pyIndPenSim {sim.mean():,.0f} ± {sim.std(ddof=1):,.0f} ({len(sim)})   "
              f"{diff:+,.0f} kg ({100 * diff / rel.mean():+.2f} %)   d {cohen_d(rel, sim):+.3f}   "
              f"KS p {ks_2samp(rel, sim).pvalue:.3f}   MWU p {mannwhitneyu(rel, sim).pvalue:.3f}")
