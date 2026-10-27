"""Run one batch with and without Raman-based PAA control and print the net yield.

    python examples/run_batch.py --seed 2000
"""
import argparse, contextlib, io
import numpy as np
from pyindpensim import indpensim_run, batch_flags, net_yield_kg

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2000)
a = ap.parse_args()

for raman, name in ((0, "Raman off"), (2, "Raman-based PAA control")):
    with contextlib.redirect_stdout(io.StringIO()):
        X = indpensim_run(1, batch_flags(raman=raman), rng=np.random.default_rng(a.seed), rtol=1e-6, atol=1e-8)
    print(f"{name:<24s} seed {a.seed}: net yield {net_yield_kg(X):,.0f} kg, batch length {X.Stats.Batch_length:.1f} h, "
          f"peak PAA {np.nanmax(X.PAA.y):,.0f} mg/L")
