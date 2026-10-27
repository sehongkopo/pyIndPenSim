# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""indpensim.py

Python port of ``indpensim.m`` — the main simulation driver.

At each sample it (1) asks the controller ``f_input`` for the manipulated
variables, (2) integrates the ODE system (:func:`indpensim_ode.indpensim_ode`)
over one sample interval, (3) stores the endpoint into the batch channels, and
(4) optionally simulates Raman spectra / off-line measurements.

Solver mapping (MATLAB -> scipy ``solve_ivp``):
    solv == 1  ode45   -> 'RK45'
    solv == 2  ode15s  -> 'BDF'    (stiff; the default used by the production run)
    solv == 3  ode23t  -> 'LSODA'

Indexing: ``k`` is the 1-based MATLAB sample counter; channel element
``X.foo.y(k)`` -> ``X.foo.y[k-1]``.

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from .create_batch import create_batch
from .indpensim_ode import indpensim_ode
from .raman_sim import raman_sim
from .substrate_prediction import substrate_prediction

_SOLVER = {1: "RK45", 2: "BDF", 3: "LSODA", 4: "Radau"}


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def indpensim(f_input, Xd, x0, h, T, solv, p, Ctrl_flags, rng=None,
              rtol=1e-3, atol=1e-6):
    """Run one IndPenSim batch.

    Parameters
    ----------
    f_input : callable
        Controller ``(X, Xd, k, h, T, Ctrl_flags) -> (u, X)``.
    Xd : object
        Disturbance / industrial-data structure (channels ``distMuP`` ...).
    x0 : object
        Initial-condition container (attributes ``S, DO2, O2, ...``).
    h : float
        Sampling period (hours).
    T : float
        Batch length (hours); ``N = T/h`` samples.
    solv : int
        Solver selector (see module docstring).
    p : ndarray
        Parameter vector (105 entries); may be modified in-place during the
        run (the mu_X cap), matching the MATLAB behaviour.
    Ctrl_flags : object
        Simulation flags (attributes ``Inhib, Dis, Vis, Raman_spec`` ...).
    rng : numpy.random.Generator, optional
        Random source passed to the Raman simulator.

    Returns
    -------
    Batch
        The populated batch structure ``X``.
    """
    if rng is None:
        rng = np.random.default_rng()
    Ctrl_flags.rng = rng   # the PRBS perturbations in fctrl use the same generator (reproducibility)

    p = np.array(p, dtype=float)  # local, mutable copy (MATLAB modifies p(2))

    N = int(round(T / h))          # experiment length in samples
    h_ode = h / 20.0               # ode solver step size (hours)
    t = np.arange(0, T + h / 2, h)  # time vector 0:h:T (length N+1)

    X = create_batch(h, T)

    # convert initial pH -> [H+]
    x0_pH_Hplus = 10 ** (-_get(x0, "pH"))

    method = _SOLVER.get(solv, "BDF")
    off_line_m = Ctrl_flags.Off_line_m
    delay = Ctrl_flags.Off_line_delay

    for k in range(1, N + 1):
        # seed initial conditions so the controller has a first input
        if k == 1:
            X.S.y[0] = _get(x0, "S")
            X.DO2.y[0] = _get(x0, "DO2")
            X.X.y[0] = _get(x0, "X")
            X.P.y[0] = _get(x0, "P")
            X.V.y[0] = _get(x0, "V")
            X.CO2outgas.y[0] = _get(x0, "CO2outgas")
            X.pH.y[0] = x0_pH_Hplus
            X.T.y[0] = _get(x0, "T")

        # get manipulated variables
        u, X = f_input(X, Xd, k, h, T, Ctrl_flags)

        # build ODE initial condition vector (33 states)
        if k == 1:
            x00 = np.array([
                _get(x0, "S"), _get(x0, "DO2"), _get(x0, "O2"), _get(x0, "P"),
                _get(x0, "V"), _get(x0, "Wt"), x0_pH_Hplus, _get(x0, "T"),
                0, 4, _get(x0, "Culture_age"),
                _get(x0, "a0"), _get(x0, "a1"), _get(x0, "a3"), _get(x0, "a4"),
                *([0] * 12),
                _get(x0, "CO2outgas"), 0, _get(x0, "PAA"), _get(x0, "NH3"), 0, 0,
            ], dtype=float)
        else:
            j = k - 2  # 0-based index of sample (k-1)
            x00 = np.array([
                X.S.y[j], X.DO2.y[j], X.O2.y[j], X.P.y[j], X.V.y[j], X.Wt.y[j],
                X.pH.y[j], X.T.y[j], X.Q.y[j], X.Viscosity.y[j],
                X.Culture_age.y[j], X.a0.y[j], X.a1.y[j], X.a3.y[j], X.a4.y[j],
                X.n0.y[j], X.n1.y[j], X.n2.y[j], X.n3.y[j], X.n4.y[j], X.n5.y[j],
                X.n6.y[j], X.n7.y[j], X.n8.y[j], X.n9.y[j], X.nm.y[j], X.phi0.y[j],
                X.CO2outgas.y[j], X.CO2_d.y[j], X.PAA.y[j], X.NH3.y[j], 0, 0,
            ], dtype=float)

        # process disturbances at this sample
        distMuP = Xd.distMuP.y[k - 1]
        distMuX = Xd.distMuX.y[k - 1]
        distcs = Xd.distcs.y[k - 1]
        distcoil = Xd.distcoil.y[k - 1]
        distabc = Xd.distabc.y[k - 1]
        distPAA = Xd.distPAA.y[k - 1]
        distTcin = Xd.distTcin.y[k - 1]
        distO_2in = Xd.distO_2in.y[k - 1]

        u00 = np.array([
            Ctrl_flags.Inhib, u["Fs"], u["Fg"], u["RPM"], u["Fc"], u["Fh"],
            u["Fb"], u["Fa"], h_ode, u["Fw"], u["pressure"], u["viscosity"],
            u["Fremoved"], u["Fpaa"], u["Foil"], u["NH3_shots"], Ctrl_flags.Dis,
            distMuP, distMuX, distcs, distcoil, distabc, distPAA, distTcin,
            distO_2in, Ctrl_flags.Vis,
        ], dtype=float)

        # mu_X cap: if biomass growth rate has fallen for a long window, clamp
        if Ctrl_flags.Inhib in (1, 2):
            if k > 65:
                seg = X.mu_X_calc.y[k - 66:k - 1]  # (k-65:k-1) 1-based
                a1 = np.diff(seg)
                a2 = a1 < 0
                if np.sum(a2) >= 63:
                    p[1] = X.mu_X_calc.y[k - 2] * 5  # p(2) = mu_X_calc(k-1)*5

        # integrate over one sample interval [t(k), t(k+1)]
        t_start = t[k - 1]
        t_end = t[k]
        sol = solve_ivp(
            lambda tt, yy: indpensim_ode(tt, yy, u00, p),
            (t_start, t_end), x00, method=method, t_eval=[t_end],
            rtol=rtol, atol=atol,
        )
        if not sol.success or sol.y.shape[1] == 0:
            raise RuntimeError(f"solve_ivp failed @k={k} (t={t_start:.2f}h): {sol.message} | "
                               f"P={x00[3]:.3f} V={x00[4]:.0f} S={x00[0]:.3f}")
        y_end = sol.y[:, -1].copy()

        # numerical stability floor on states 1..31 (0-based 0..30)
        for n in range(31):
            if y_end[n] <= 0:
                y_end[n] = 0.001

        # ---- optional recording hook (does not change the simulation) --------
        _rec = getattr(Ctrl_flags, "_rec", None)
        if _rec is not None:
            _rec.setdefault("x00", []).append(x00.copy())   # state at the start of the step (solver input)
            _rec.setdefault("u", []).append(u00.copy())     # input vector (26)
            _rec.setdefault("par", []).append(p.copy())     # parameter vector (105), after caps
            _rec.setdefault("y", []).append(y_end.copy())   # state at the end of the step (after flooring at 0)
            _rec.setdefault("t", []).append((t_start, t_end))

        # ---- store manipulated variables ---------------------------------
        for name, val in (
            ("Fg", u["Fg"]), ("RPM", u["RPM"]), ("Fpaa", u["Fpaa"]),
            ("Fs", u["Fs"]), ("Fa", u["Fa"]), ("Fb", u["Fb"]), ("Fc", u["Fc"]),
            ("Foil", u["Foil"]), ("Fh", u["Fh"]), ("Fw", u["Fw"]),
            ("pressure", u["pressure"]), ("Fremoved", u["Fremoved"]),
        ):
            ch = getattr(X, name)
            ch.t[k - 1] = t_end
            ch.y[k - 1] = val

        # ---- store IndPenSim states --------------------------------------
        X.S.y[k - 1] = y_end[0]; X.S.t[k - 1] = t_end
        X.DO2.y[k - 1] = y_end[1]
        X.DO2.y[k - 1] = 1 if X.DO2.y[k - 1] < 2 else X.DO2.y[k - 1]
        X.DO2.t[k - 1] = t_end
        X.O2.y[k - 1] = y_end[2]; X.O2.t[k - 1] = t_end
        X.P.y[k - 1] = y_end[3]; X.P.t[k - 1] = t_end
        X.V.y[k - 1] = y_end[4]; X.V.t[k - 1] = t_end
        X.Wt.y[k - 1] = y_end[5]; X.Wt.t[k - 1] = t_end
        X.pH.y[k - 1] = y_end[6]; X.pH.t[k - 1] = t_end
        X.T.y[k - 1] = y_end[7]; X.T.t[k - 1] = t_end
        X.Q.y[k - 1] = y_end[8]; X.Q.t[k - 1] = t_end
        X.Viscosity.y[k - 1] = y_end[9]; X.Viscosity.t[k - 1] = t_end
        X.Culture_age.y[k - 1] = y_end[10]; X.Culture_age.t[k - 1] = t_end
        X.a0.y[k - 1] = y_end[11]; X.a0.t[k - 1] = t_end
        X.a1.y[k - 1] = y_end[12]; X.a1.t[k - 1] = t_end
        X.a3.y[k - 1] = y_end[13]; X.a3.t[k - 1] = t_end
        X.a4.y[k - 1] = y_end[14]; X.a4.t[k - 1] = t_end
        X.n0.y[k - 1] = y_end[15]; X.n0.t[k - 1] = t_end
        X.n1.y[k - 1] = y_end[16]; X.n1.t[k - 1] = t_end
        X.n2.y[k - 1] = y_end[17]; X.n2.t[k - 1] = t_end
        X.n3.y[k - 1] = y_end[18]; X.n3.t[k - 1] = t_end
        X.n4.y[k - 1] = y_end[19]; X.n4.t[k - 1] = t_end
        X.n5.y[k - 1] = y_end[20]; X.n5.t[k - 1] = t_end
        X.n6.y[k - 1] = y_end[21]; X.n6.t[k - 1] = t_end
        X.n7.y[k - 1] = y_end[22]; X.n7.t[k - 1] = t_end
        X.n8.y[k - 1] = y_end[23]; X.n8.t[k - 1] = t_end
        X.n9.y[k - 1] = y_end[24]; X.n9.t[k - 1] = t_end
        X.nm.y[k - 1] = y_end[25]; X.nm.t[k - 1] = t_end
        X.phi0.y[k - 1] = y_end[26]; X.phi0.t[k - 1] = t_end
        X.CO2outgas.y[k - 1] = y_end[27]; X.CO2outgas.t[k - 1] = t_end
        X.CO2_d.t[k - 1] = t_end; X.CO2_d.y[k - 1] = y_end[28]
        X.PAA.y[k - 1] = y_end[29]; X.PAA.t[k - 1] = t_end
        X.NH3.y[k - 1] = y_end[30]; X.NH3.t[k - 1] = t_end
        X.mu_P_calc.y[k - 1] = y_end[31]; X.mu_P_calc.t[k - 1] = t_end
        X.mu_X_calc.y[k - 1] = y_end[32]; X.mu_X_calc.t[k - 1] = t_end

        X.X.y[k - 1] = X.a0.y[k - 1] + X.a1.y[k - 1] + X.a3.y[k - 1] + X.a4.y[k - 1]
        X.X.t[k - 1] = t_end
        X.Fault_ref.y[k - 1] = u["Fault_ref"]; X.Fault_ref.t[k - 1] = t_end
        X.Control_ref.y[k - 1] = Ctrl_flags.PRBS
        X.Control_ref.t[k - 1] = Ctrl_flags.Batch_Num
        X.PAT_ref.y[k - 1] = Ctrl_flags.Raman_spec
        X.PAT_ref.t[k - 1] = Ctrl_flags.Batch_Num
        X.Batch_ref.t[k - 1] = Ctrl_flags.Batch_Num
        X.Batch_ref.y[k - 1] = Ctrl_flags.Batch_Num

        # ---- OUR / CER ----------------------------------------------------
        O2_in = 0.204  # % oxygen in air (local; distinct from par O_2_in)
        X.OUR.y[k - 1] = (32 * X.Fg.y[k - 1] / 22.4) * (
            O2_in - X.O2.y[k - 1] * (0.7902 / (1 - X.O2.y[k - 1] - X.CO2outgas.y[k - 1] / 100)))
        X.OUR.t[k - 1] = t_end
        X.CER.y[k - 1] = (44 * X.Fg.y[k - 1] / 22.4) * (
            (0.65 * X.CO2outgas.y[k - 1] / 100) * (
                0.7902 / (1 - O2_in - X.CO2outgas.y[k - 1] / 100) - 0.0330))
        X.CER.t[k - 1] = t_end

        # ---- Raman spectra ------------------------------------------------
        if k > 10:
            if Ctrl_flags.Raman_spec == 1:
                X = raman_sim(k, X, h, T, rng=rng)
            elif Ctrl_flags.Raman_spec == 2:
                X = raman_sim(k, X, h, T, rng=rng)
                X = substrate_prediction(k, X, h, T)

        # ---- off-line measurements ---------------------------------------
        rem = t_end % off_line_m
        is_sample_time = (
            math.isclose(rem, 0.0, abs_tol=1e-6)
            or math.isclose(rem, off_line_m, abs_tol=1e-6)
            or math.isclose(t_end, 1.0, abs_tol=1e-6)
            or math.isclose(t_end, T, abs_tol=1e-6)
        )
        if is_sample_time and (k - delay) >= 1:
            d = int(delay)
            X.NH3_offline.y[k - 1] = X.NH3.y[k - 1 - d]
            X.NH3_offline.t[k - 1] = X.NH3.t[k - 1 - d]
            X.Viscosity_offline.y[k - 1] = X.Viscosity.y[k - 1 - d]
            X.Viscosity_offline.t[k - 1] = X.Viscosity.t[k - 1 - d]
            X.PAA_offline.y[k - 1] = X.PAA.y[k - 1 - d]
            X.PAA_offline.t[k - 1] = X.PAA.t[k - 1 - d]
            X.P_offline.y[k - 1] = X.P.y[k - 1 - d]
            X.P_offline.t[k - 1] = X.P.t[k - 1 - d]
            X.X_offline.y[k - 1] = X.X.y[k - 1 - d]
            X.X_offline.t[k - 1] = X.X.t[k - 1 - d]
        else:
            X.NH3_offline.y[k - 1] = np.nan; X.NH3_offline.t[k - 1] = np.nan
            X.Viscosity_offline.y[k - 1] = np.nan; X.Viscosity_offline.t[k - 1] = np.nan
            X.PAA_offline.y[k - 1] = np.nan; X.PAA_offline.t[k - 1] = np.nan
            X.P_offline.y[k - 1] = np.nan; X.P_offline.t[k - 1] = np.nan
            X.X_offline.y[k - 1] = np.nan; X.X_offline.t[k - 1] = np.nan

    # ---- unit conversions -------------------------------------------------
    X.pH.y = -np.log10(X.pH.y)   # [H+] -> pH
    X.Q.y = X.Q.y / 1000          # Qrxn -> kcal

    return X
