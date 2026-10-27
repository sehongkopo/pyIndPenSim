# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""pyIndPenSim: a Python port of the IndPenSim industrial-scale penicillin simulator.

Each module mirrors one MATLAB ``.m`` file of IndPenSim V2.02
(Goldrick et al., 2015, 2019). The stiff solver ``ode15s`` is replaced by the
BDF method of :func:`scipy.integrate.solve_ivp`.

Quick start
-----------
>>> import numpy as np
>>> from pyindpensim import indpensim_run, batch_flags
>>> X = indpensim_run(1, batch_flags(), rng=np.random.default_rng(2000), rtol=1e-6, atol=1e-8)
>>> X.Stats.Penicllin_yield_total / 1000   # net penicillin yield, kg
"""
from types import SimpleNamespace as _NS

from .indpensim_run import indpensim_run, prepare_batch
from .parameter_list import parameter_list

__version__ = "1.0.0"
__all__ = ["indpensim_run", "prepare_batch", "parameter_list", "batch_flags", "net_yield_kg", "__version__"]


def batch_flags(control_strategy=1, variable_length=1, fault=0, raman=0, **extra):
    """Run flags for a single batch (``Batch_no = 1``).

    control_strategy : 0 recipe-driven (sequential batch control), 1 operator-like PRBS perturbations.
    variable_length  : 0 fixed batch length, 1 uneven batch length.
    fault            : fault reference of the released code (0 = no fault).
    raman            : 0 no Raman, 1 record Raman spectra, 2 Raman-based PAA control.
    extra            : optional flags of the extensions (e.g. ``PAA_setpoint_ctrl=True``, ``PAA_sp=1200.0``).
    """
    return _NS(Control_strategy=[control_strategy], Batch_length=[variable_length],
               Batch_fault_order_reference=[fault], Raman_spec=[raman], **extra)


def net_yield_kg(X):
    """Net penicillin yield of a finished batch in kg (harvest at the end plus the amount discharged)."""
    return X.Stats.Penicllin_yield_total / 1000.0
