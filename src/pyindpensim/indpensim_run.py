# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""indpensim_run.py

Python port of ``indpensim_run.m`` — the per-batch wrapper.

It sets the control flags, draws randomised initial conditions, builds the
low-pass-filtered process disturbances, assembles the parameter vector and
calls :func:`indpensim.indpensim`, then computes penicillin-yield statistics.

Random numbers
--------------
The MATLAB source reseeds the global generator before each draw
(``rng(Seed_ref + Batch_no + Rand_ref)``). We reproduce the *structure* with
per-draw ``numpy`` generators, but the exact numeric stream differs from
MATLAB's ``randn`` — so individual batches will not be bit-identical to the
MATLAB output (the statistical behaviour is preserved).

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

from types import SimpleNamespace

import os as _os
import numpy as np
from scipy.signal import lfilter

from .create_channel import create_channel
from .fctrl_indpensim import fctrl_indpensim
from .indpensim import indpensim
from .parameter_list import parameter_list


def _flag(batch_run_flags, name, batch_no):
    """Read ``batch_run_flags.<name>[batch_no-1]`` (1-based batch index)."""
    seq = batch_run_flags[name] if isinstance(batch_run_flags, dict) \
        else getattr(batch_run_flags, name)
    return np.asarray(seq).reshape(-1)[batch_no - 1]


def prepare_batch(Batch_no, Batch_run_flags, rng, pH_sp=6.5, T_sp=298.0, lever=None, _rec=None):
    """Set up a batch (flags, random initial conditions, disturbances, parameters) and return (Xinterp, x0, h, T, par, Ctrl_flags)."""
    # ================= simulation flags =================
    Ctrl_flags = SimpleNamespace()
    Ctrl_flags.SBC = 0
    Ctrl_flags.PRBS = _flag(Batch_run_flags, "Control_strategy", Batch_no)
    Ctrl_flags.Fixed_Batch_length = _flag(Batch_run_flags, "Batch_length", Batch_no)
    Ctrl_flags.IC = 0
    Ctrl_flags.Inhib = 2
    Ctrl_flags.Dis = 1
    Ctrl_flags.Faults = _flag(Batch_run_flags, "Batch_fault_order_reference", Batch_no)
    Ctrl_flags.Vis = 0
    Ctrl_flags.Raman_spec = _flag(Batch_run_flags, "Raman_spec", Batch_no)
    Ctrl_flags.Batch_Num = Batch_no
    Ctrl_flags.Off_line_m = 12       # off-line sampling rate (hours)
    Ctrl_flags.Off_line_delay = 4    # off-line analysis delay (hours)
    Ctrl_flags.plots = 1
    # Flags of the optional PAA set-point PI control (APC of Goldrick et al., 2019)
    Ctrl_flags.PAA_setpoint_ctrl = getattr(Batch_run_flags, "PAA_setpoint_ctrl", False)
    Ctrl_flags.PAA_sp = getattr(Batch_run_flags, "PAA_sp", 1250.0)
    Ctrl_flags.PAA_ctrl_start_h = getattr(Batch_run_flags, "PAA_ctrl_start_h", 25.0)
    Ctrl_flags.PAA_ctrl_use_pred = getattr(Batch_run_flags, "PAA_ctrl_use_pred", False)  # feed back a PAA reading with measurement error
    Ctrl_flags.PAA_ctrl_real_pls = getattr(Batch_run_flags, "PAA_ctrl_real_pls", False)  # feed back a recalibrated PLS prediction
    Ctrl_flags.PAA_pred_scale = getattr(Batch_run_flags, "PAA_pred_scale", 1000.0)  # unit conversion of PAA_pred.
    # Optional PI on an estimated PAA: estimator function, second set-point, gain multiplier and output limit. Unset = released behaviour.
    Ctrl_flags.PAA_est_fn = getattr(Batch_run_flags, "PAA_est_fn", None)
    Ctrl_flags.PAA_sp2 = getattr(Batch_run_flags, "PAA_sp2", None)
    Ctrl_flags.PAA_t_sw = getattr(Batch_run_flags, "PAA_t_sw", 1e9)
    Ctrl_flags.PAA_kp_mult = getattr(Batch_run_flags, "PAA_kp_mult", 1.0)
    Ctrl_flags.PAA_umax = getattr(Batch_run_flags, "PAA_umax", 150.0)
    #   Default 1000 matches PAA_PLS_model.mat (g/L); pass 1.0 for a PLS recalibrated in mg/L.

    x0 = SimpleNamespace()

    # ============ randomised initial conditions ============
    if Ctrl_flags.IC == 0:
        Ctrl_flags.SBC = 0
        Ctrl_flags.Vis = 0
        Optimum_Batch_length = 230  # standard batch length (h)
        if Ctrl_flags.Fixed_Batch_length == 1:
            Batch_length_variation = 25 * rng.standard_normal()
            T = Optimum_Batch_length + Batch_length_variation
            T = int(round(T))
        else:  # Fixed_Batch_length == 0
            T = Optimum_Batch_length

        Randomise_each_batch = 1
        if Randomise_each_batch == 1:
            Random_seed_ref = int(np.ceil(rng.random() * 1000))
        else:
            Random_seed_ref = 5

        Seed_ref = 31 + Random_seed_ref
        rand_ref = 1

        def draw():
            nonlocal rand_ref
            seed = int(Seed_ref + Batch_no + rand_ref)
            rand_ref += 1
            return np.random.default_rng(seed).standard_normal()

        intial_conds = 0.5 + 0.05 * draw()        # X_0
        #  Optional: widen the batch-to-batch spread of the strain parameters with the
        #    MU_SIGMA_MULT environment variable. These two lines set the spread of par[0] = mu_p
        #    and par[1] = mux_max (parameter_list.py), separate from the time-varying disturbances.
        #    Unset (1.0) leaves the released behaviour unchanged; only the standard deviation is
        #    scaled, never the mean.
        _msm = float(_os.environ.get('MU_SIGMA_MULT', '1.0'))
        x0.mux = 0.41 + 0.025 * _msm * draw()     # max biomass growth rate
        x0.mup = 0.041 + 0.0025 * _msm * draw()   # max penicillin growth rate
        h = 0.2                                    # sampling rate 12 min
        x0.S = 1 + 0.1 * draw()                    # substrate [g/L]
        x0.DO2 = 15 + 0.5 * draw()                 # dissolved O2 [mg/L]
        x0.X = intial_conds + 0.1 * draw()         # biomass [g/L]
        x0.P = 0                                   # penicillin [g/L]
        x0.V = 5.800e+04 + 500 * draw()            # volume [L]
        x0.Wt = 6.2e+04 + 500 * draw()             # weight [kg]
        x0.CO2outgas = 0.038 + 0.001 * draw()      # CO2 off-gas [%]
        x0.O2 = 0.20 + 0.05 * draw()               # O2 off-gas [%]
        x0.pH = 6.5 + 0.1 * draw()                 # pH
        x0.T = 297 + 0.5 * draw()                  # temperature [K]
        x0.a0 = intial_conds * (1 / 3)             # a0 biomass [g/L]
        x0.a1 = intial_conds * (2 / 3)             # a1 biomass [g/L]
        x0.a3 = 0
        x0.a4 = 0
        x0.Culture_age = 0
        x0.PAA = 1400 + 50 * draw()                # PAA [mg/L]
        x0.NH3 = 1700 + 50 * draw()                # nitrogen [mg/L]
        alpha_kla = 85 + 10 * draw()               # kla constant
        PAA_c = 530000 + 20000 * draw()            # PAA conc in F_PAA [mg/L]
        N_conc_paa = 2 * 75000 + 2000 * draw()     # N conc in F_PAA [mg/L]
        Batch_time = np.arange(0, T + h / 2, h)

        # Temperature and pH set-points (may be set by an operator or optimiser)
        Ctrl_flags.T_sp = T_sp
        Ctrl_flags.pH_sp = pH_sp
        Ctrl_flags.lever = lever   # {operating variable: multiplier (scalar/array)}, applied to u in fctrl
    Ctrl_flags._rec = _rec         # optional dict that records (u, par, y) at every step (None = no recording)

    # ============ process disturbances (low-pass filtered) ============
    N = int(round(T / h))
    dist_rng = np.random.default_rng(int(Random_seed_ref + Batch_no))
    b1 = 1 - 0.995
    a1 = [1, -0.995]

    def dist(scale):
        v = dist_rng.standard_normal(N + 1)  # randn(T/h+1, 1)
        return lfilter([b1], a1, scale * v)

    Xinterp = SimpleNamespace()
    Xinterp.distMuP = create_channel('Penicillin specific growth rate disturbance', 'g/Lh', 'h', Batch_time, dist(0.03))
    Xinterp.distMuX = create_channel('Biomass specific  growth rate disturbance', 'hr^{-1}', 'h', Batch_time, dist(0.25))
    Xinterp.distcs = create_channel('Substrate concentration disturbance ', ' g L^{-1}', 'h', Batch_time, dist(5 * 300))
    Xinterp.distcoil = create_channel('Substrate concentration disturbance ', ' g L^{-1}', 'h', Batch_time, dist(300))
    Xinterp.distabc = create_channel('Acid/Base concentration disturbance ', ' g L^{-1}', 'h', Batch_time, dist(0.2))
    # MATLAB computes distPAA twice (second overwrites); replicated for parity.
    _ = dist(300000)
    Xinterp.distPAA = create_channel('Phenylacetic acid concentration disturbance ', ' g L^{-1}', 'h', Batch_time, dist(300000))
    Xinterp.distTcin = create_channel('Coolant inlet temperature disturbance ', 'K', 'h', Batch_time, dist(100))
    Xinterp.distO_2in = create_channel('Oxygen inlet concentration', '%', 'h', Batch_time, dist(0.02))

    # Optional measurement-error series for the PAA reading: measured = true x (1 + e), e a relative error.
    #   Used instead of the built-in PLS prediction when a simple error model is wanted.
    if getattr(Ctrl_flags, "PAA_ctrl_use_pred", False):
        sigma = float(getattr(Batch_run_flags, "PAA_sensor_sigma", 0.10))
        bias = float(getattr(Batch_run_flags, "PAA_sensor_bias", 0.0))
        model = str(getattr(Batch_run_flags, "PAA_sensor_model", "ar"))   # 'ar'|'white'|'white_ma3'
        se_rng = np.random.default_rng(int(Random_seed_ref + Batch_no + 777))
        e = se_rng.standard_normal(N + 2)
        if model == "white_ma3":                   # white error then a 3-point moving average, as in substrate_prediction
             eps = np.copy(e); eps[2:] = (e[2:] + e[1:-1] + e[:-2]) / 3.0
        elif model == "white":                     # white error without averaging
            eps = e
        else:                                      # AR(1) correlated error (default)
            a = 0.85; eps = lfilter([np.sqrt(1 - a * a)], [1, -a], e)
        Ctrl_flags.paa_sensor_err = 1.0 + bias + sigma * eps

    # ============ parameter list & simulation ============
    par = parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)
    return Xinterp, x0, h, T, par, Ctrl_flags


def indpensim_run(Batch_no, Batch_run_flags, rng=None, rtol=1e-3, atol=1e-6, solv=2,
                  pH_sp=6.5, T_sp=298.0, lever=None, _rec=None):
    """Run a single batch and return the populated batch structure ``Xref``.

    Parameters
    ----------
    Batch_no : int
        1-based batch index.
    Batch_run_flags : object or dict
        Must provide the 1-based-indexable sequences ``Control_strategy``,
        ``Batch_length``, ``Batch_fault_order_reference`` and ``Raman_spec``.
    rng : numpy.random.Generator, optional
        Generator for the *non-seeded* global draws (batch-length variation and
        the master random seed). Defaults to a fresh ``default_rng()``.
    """
    if rng is None:
        rng = np.random.default_rng()

    Xinterp, x0, h, T, par, Ctrl_flags = prepare_batch(
        Batch_no, Batch_run_flags, rng, pH_sp=pH_sp, T_sp=T_sp, lever=lever, _rec=_rec)


    print('Running IndPenSim...')
    Xref = indpensim(fctrl_indpensim, Xinterp, x0, h, T, solv, par, Ctrl_flags,
                     rng=rng, rtol=rtol, atol=atol)

    # ============ penicillin-yield statistics ============
    Stats = SimpleNamespace()
    Stats.Penicllin_harvested_during_batch = float(np.sum(Xref.Fremoved.y * Xref.P.y) * h)
    Stats.Penicllin_harvested_end_of_batch = float(Xref.V.y[-1] * Xref.P.y[-1])
    Stats.Penicllin_yield_total = Stats.Penicllin_harvested_end_of_batch - Stats.Penicllin_harvested_during_batch
    Stats.Batch_length = float(Xref.V.t[-1])
    Xref.Stats = Stats

    print(f"Penicillin harvested during the batch {round(Stats.Penicllin_harvested_during_batch / 1000)} Kg")
    print(f"Final Penicillin yield at harvest {round(Stats.Penicllin_harvested_end_of_batch / 1000)} Kg")
    print(f"Total penicillin {round(Stats.Penicllin_yield_total / 1000)} Kg")

    return Xref


if __name__ == "__main__":
    # Minimal smoke test: one recipe-driven batch, no faults, no Raman.
    flags = SimpleNamespace(
        Control_strategy=[0],
        Batch_length=[0],
        Batch_fault_order_reference=[0],
        Raman_spec=[0],
    )
    Xref = indpensim_run(1, flags, rng=np.random.default_rng(0))
    print("Batch length (h):", Xref.Stats.Batch_length)
