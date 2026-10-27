import contextlib, io
import numpy as np
from pyindpensim import indpensim_run, batch_flags, net_yield_kg

# Yields of seed 2000 recorded with the version used in the accompanying study.
EXPECTED = {0: 2334.68, 2: 4029.68}


def _run(raman):
    with contextlib.redirect_stdout(io.StringIO()):
        X = indpensim_run(1, batch_flags(raman=raman), rng=np.random.default_rng(2000), rtol=1e-6, atol=1e-8, solv=2)
    return net_yield_kg(X)


def test_raman_off_reproduces_reference():
    assert abs(_run(0) - EXPECTED[0]) < 0.5


def test_raman_on_reproduces_reference():
    assert abs(_run(2) - EXPECTED[2]) < 0.5
