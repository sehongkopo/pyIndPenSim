# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""fctrl_indpensim.py

Python port of ``fctrl_indpensim.m``.

Computes the manipulated variables (control actions) for one sample of the
IndPenSim run: pH control (acid/base PID), temperature control (heat/cool
PID), the Sequential Batch Control (SBC) recipes for the feeds, injected
process faults, and — when Raman-based control is active — a PID on PAA.

Indexing convention
-------------------
``k`` is kept as the **1-based** MATLAB sample counter. Any channel access
``X.foo.y(idx)`` in MATLAB becomes ``X.foo.y[idx-1]`` here.

The function returns ``(u, X)`` where ``u`` is a dict of manipulated variables.

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

import numpy as np

from .pid_simple3 import pid_simple3


def _recipe_lookup(k, thresholds, setpoints):
    """MATLAB recipe loop: first ``setpoint`` whose ``k <= threshold`` wins.

    Falls back to the last setpoint if ``k`` exceeds every threshold.
    """
    for thr, sp in zip(thresholds, setpoints):
        if k <= thr:
            return sp
    return setpoints[-1]


def _ensure_prbs(X, N):
    """Ensure ``X.PRBS_noise_addition`` exists as a length-N zero array."""
    if not hasattr(X, "PRBS_noise_addition") or X.PRBS_noise_addition is None:
        X.PRBS_noise_addition = np.zeros(N)
    return X.PRBS_noise_addition


def _ph(hplus):
    """Convert an [H+] concentration to pH ( -log10 )."""
    return -np.log10(hplus)


def fctrl_indpensim(X, Xd, k, h, T, Ctrl_flags):
    """Return ``(u, X)`` — the manipulated variables for sample ``k`` (1-based)."""
    N = int(round(T / h))
    u = {}

    # =====================================================================
    #  pH controller
    # =====================================================================
    pH_sensor_error = 0.0
    if Ctrl_flags.Faults == 8:
        pH_sensor_error = 0.1
        xp = [0, 200, 800, 1750]
        fp = [0, 0, pH_sensor_error, pH_sensor_error]
        pH_sensor_error = float(np.interp(k, xp, fp))
        u["Fault_ref"] = 1

    # pH set-point: a scalar or a per-step array (time-varying set-point)
    pH_sp = Ctrl_flags.pH_sp
    if hasattr(pH_sp, "__len__"):
        pH_sp = pH_sp[min(k - 1, len(pH_sp) - 1)]
    if k == 1 or k == 2:
        ph_err = pH_sp - _ph(X.pH.y[0]) + pH_sensor_error
        ph_err1 = pH_sp - _ph(X.pH.y[0]) + pH_sensor_error
    else:
        ph_err = pH_sp - _ph(X.pH.y[k - 2]) + pH_sensor_error
        # NB: MATLAB omits pH_sp in ph_err1 here — preserved faithfully.
        ph_err1 = -_ph(X.pH.y[k - 3]) + pH_sensor_error

    if k == 1 or k == 2:
        ph = _ph(X.pH.y[0]); ph1 = _ph(X.pH.y[0]); ph2 = _ph(X.pH.y[0])
    elif k == 3:
        ph = _ph(X.pH.y[1]); ph1 = _ph(X.pH.y[0]); ph2 = _ph(X.pH.y[0])
    else:
        ph = _ph(X.pH.y[k - 2]); ph1 = _ph(X.pH.y[k - 3]); ph2 = _ph(X.pH.y[k - 4])

    if ph_err >= -0.05:
        ph_on_off = 1
        if k == 1:
            Fb = pid_simple3(X.Fb.y[0], ph_err, ph_err1, ph, ph1, ph2,
                             0, 225, 8e-2, 4.0000e-05, 8, h)
        else:
            Fb = pid_simple3(X.Fb.y[k - 2], ph_err, ph_err1, ph, ph1, ph2,
                             0, 225, 8e-2, 4.0000e-05, 8, h)
        Fa = 0.0
    elif ph_err <= -0.05:
        ph_on_off = 1
        if k == 1:
            Fa = pid_simple3(X.Fa.y[0], ph_err, ph_err1, ph, ph1, ph2,
                             0, 225, 8e-2, 12.5, 0.125, h)
            Fb = 0.0
        else:
            Fa = pid_simple3(X.Fa.y[k - 2], ph_err, ph_err1, ph, ph1, ph2,
                             0, 225, 8e-2, 12.5, 0.125, h)
            Fb = X.Fb.y[k - 2] * 0.5
    else:
        ph_on_off = 0
        Fb = 0.0
        Fa = 0.0

    # =====================================================================
    #  Temperature controller
    # =====================================================================
    T_sensor_error = 0.0
    if Ctrl_flags.Faults == 7:
        T_sensor_error = 0.4
        xp = [0, 200, 800, 1750]
        fp = [0, 0, T_sensor_error, T_sensor_error]
        T_sensor_error = float(np.interp(k, xp, fp))
        u["Fault_ref"] = 1

    T_sp = Ctrl_flags.T_sp
    if hasattr(T_sp, "__len__"):
        T_sp = T_sp[min(k - 1, len(T_sp) - 1)]
    if k == 1 or k == 2:
        temp_err = T_sp - X.T.y[0] + T_sensor_error
        temp_err1 = T_sp - X.T.y[0] + T_sensor_error
    else:
        temp_err = T_sp - X.T.y[k - 2] + T_sensor_error
        temp_err1 = T_sp - X.T.y[k - 3] + T_sensor_error

    if k == 1 or k == 2:
        temp = X.T.y[0]; temp1 = X.T.y[0]; temp2 = X.T.y[0]
    elif k == 3:
        temp = X.T.y[1]; temp1 = X.T.y[0]; temp2 = X.T.y[0]
    else:
        temp = X.T.y[k - 2]; temp1 = X.T.y[k - 3]; temp2 = X.T.y[k - 4]

    if temp_err <= 0.05:
        temp_on_off = 0  # cooling
        if k == 1:
            Fc = pid_simple3(X.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2,
                             0, 1.5e3, -300, 1.6, 0.005, h)
            Fh = 0.0
        else:
            Fc = pid_simple3(X.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2,
                             0, 1.5e3, -300, 1.6, 0.005, h)
            Fh = X.Fh.y[k - 2] * 0.1
    else:
        temp_on_off = 1  # heating (gain is negative)
        if k == 1:
            Fh = pid_simple3(X.Fc.y[0], temp_err, temp_err1, temp, temp1, temp2,
                             0, 1.5e3, 50, 0.050, 1, h)
            Fc = 0.0
        else:
            Fh = pid_simple3(X.Fc.y[k - 2], temp_err, temp_err1, temp, temp1, temp2,
                             0, 1.5e3, 50, 0.050, 1, h)
            Fc = X.Fc.y[k - 2] * 0.3

    # numerical stability floors
    if Fc < 1e-4:
        Fc = 1e-4
    if Fh < 1e-4:
        Fh = 1e-4

    # Defaults for variables that only the SBC branch sets, so they always exist.
    Foil = 0.0
    F_discharge = 0.0
    pressure = 1.0
    Fpaa = 0.0
    Fw = 0.0
    viscosity = 4.0
    Fg = 30.0
    Fs = 8.0

    # =====================================================================
    #  Sequential Batch Control strategy
    # =====================================================================
    if Ctrl_flags.SBC == 1:
        # Operator-controlled: replay recorded industrial inputs from Xd.
        Foil = Xd.Foil.y[k - 1]
        F_discharge = Xd.F_discharge_cal.y[k - 1]
        pressure = Xd.pressure.y[k - 1]
        Fpaa = Xd.Fpaa.y[k - 1]
        Fw = Xd.Fw.y[k - 1]
        viscosity = Xd.viscosity.y[k - 1]
        Fg = Xd.Fg.y[k - 1]
        Fs = Xd.Fs.y[k - 1]

    if Ctrl_flags.SBC == 0:
        viscosity = 4.0
        # ---- SBC - Fs ----
        Recipe_Fs = [15, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260,
                     280, 300, 320, 340, 360, 380, 400, 800, 1750]
        Recipe_Fs_sp = [8, 15, 30, 75, 150, 30, 37, 43, 47, 51, 57, 61, 65, 72,
                        76, 80, 84, 90, 116, 90, 80]  # adjusted
        Fs = _recipe_lookup(k, Recipe_Fs, Recipe_Fs_sp)

        prbs = _ensure_prbs(X, N)
        _prng = getattr(Ctrl_flags, "rng", None) or np.random.default_rng()
        if Ctrl_flags.PRBS == 1:
            if k > 500 and (k % 100) == 0:
                random_number = _prng.integers(1, 4)
                noise_factor = 15
                if random_number == 1:
                    random_noise = 0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                prbs[k - 1] = random_noise
            if k > 475:
                Fs = X.Fs.y[k - 2]
            if k > 500 and (k % 100) == 0:
                Fs = X.Fs.y[k - 2] + prbs[k - 1]
        else:
            prbs[k - 1] = 0

        # ---- SBC - Foil ----
        Recipe_Foil = [20, 80, 280, 300, 320, 340, 360, 380, 400, 1750]
        Recipe_Foil_sp = [22, 30, 35, 34, 33, 32, 31, 30, 29, 23]
        Foil = _recipe_lookup(k, Recipe_Foil, Recipe_Foil_sp)

        # ---- SBC - Fg ----
        Recipe_Fg = [40, 100, 200, 450, 1000, 1250, 1750]
        Recipe_Fg_sp = [30, 42, 55, 60, 75, 65, 60]
        Fg = _recipe_lookup(k, Recipe_Fg, Recipe_Fg_sp)

        # ---- SBC - pressure ----
        Recipe_pres = [62.5, 125, 150, 200, 500, 750, 1000, 1750]
        Recipe_pres_sp = [0.6, 0.7, 0.8, 0.9, 1.1, 1, 0.9, 0.9]
        pressure = _recipe_lookup(k, Recipe_pres, Recipe_pres_sp)

        # ---- SBC - F_discharge ----
        Recipe_discharge = [500, 510, 650, 660, 750, 760, 850, 860, 950, 960,
                            1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750]
        Recipe_discharge_sp = [0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000, 0,
                               4000, 0, 4000, 0, 4000, 0, 4000, 0, 0]
        # MATLAB: negates the matched value except the fall-through end value.
        matched = None
        for thr, sp in zip(Recipe_discharge, Recipe_discharge_sp):
            if k <= thr:
                matched = -sp
                break
        F_discharge = matched if matched is not None else Recipe_discharge_sp[-1]

        # ---- SBC - Fw ----
        Recipe_water = [250, 375, 750, 800, 850, 1000, 1250, 1350, 1750]
        Recipe_water_sp = [0, 500, 100, 0, 400, 150, 250, 0, 100]
        Fw = _recipe_lookup(k, Recipe_water, Recipe_water_sp)

        # ---- SBC - F_PAA ----
        Recipe_PAA = [25, 200, 1000, 1500, 1750]
        Recipe_PAA_sp = [5, 0, 10, 4, 0]
        Fpaa = _recipe_lookup(k, Recipe_PAA, Recipe_PAA_sp)

        if Ctrl_flags.PRBS == 1:
            if k > 500 and (k % 100) == 0:
                random_number = _prng.integers(1, 4)
                noise_factor = 1
                if random_number == 1:
                    random_noise = 0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                prbs[k - 1] = random_noise
            if k > 475:
                Fpaa = X.Fpaa.y[k - 2]
            if k > 500 and (k % 100) == 0:
                Fpaa = X.Fpaa.y[k - 2] + prbs[k - 1]
        else:
            prbs[k - 1] = 0

        # ---- SBC - NH3_shots ----
        _ensure_channel_value(Xd, "NH3_shots", N, k, 0.0)

    # =====================================================================
    #  Process faults
    # =====================================================================
    if Ctrl_flags.Faults == 1 or Ctrl_flags.Faults == 6:  # aeration
        if 100 <= k <= 120:
            Fg = 20; u["Fault_ref"] = 1
        if 500 <= k <= 550:
            Fg = 20; u["Fault_ref"] = 1

    if Ctrl_flags.Faults == 2 or Ctrl_flags.Faults == 6:  # pressure
        if 500 <= k <= 520:
            pressure = 2; u["Fault_ref"] = 1
        if 1000 <= k <= 1200:
            pressure = 2; u["Fault_ref"] = 1

    if Ctrl_flags.Faults == 3 or Ctrl_flags.Faults == 6:  # substrate feed
        if 100 <= k <= 150:
            Fs = 2; u["Fault_ref"] = 1
        if 380 <= k <= 460:
            Fs = 20; u["Fault_ref"] = 1
        if 1000 <= k <= 1070:
            Fs = 20; u["Fault_ref"] = 1

    if Ctrl_flags.Faults == 4 or Ctrl_flags.Faults == 6:  # base flowrate
        if 400 <= k <= 420:
            Fb = 5; u["Fault_ref"] = 1
        if 700 <= k <= 800:
            Fb = 10; u["Fault_ref"] = 1

    if Ctrl_flags.Faults == 5 or Ctrl_flags.Faults == 6:  # coolant flowrate
        if 350 <= k <= 450:
            Fc = 2; u["Fault_ref"] = 1
        if 1200 <= k <= 1350:
            Fc = 10; u["Fault_ref"] = 1

    # =====================================================================
    #  Raman-based PAA controller (Raman_spec == 2)
    # =====================================================================
    if Ctrl_flags.Raman_spec == 2:
        PAA_sp = 1200
        if k == 1 or k == 2:
            PAA_err = PAA_sp - X.PAA.y[0]
            PAA_err1 = PAA_sp - X.PAA.y[0]
        else:
            PAA_err = PAA_sp - X.PAA.y[k - 2]
            PAA_err1 = PAA_sp - X.PAA.y[k - 3]

        if (k * h) < 10:
            Fpaa = Fpaa  # open loop for the first 10 hours
        else:
            if k == 1 or k == 2:
                temp_p = X.PAA_pred.y[0]; temp1_p = X.PAA_pred.y[0]; temp2_p = X.PAA_pred.y[0]
            elif k == 3:
                temp_p = X.PAA_pred.y[1]; temp1_p = X.PAA_pred.y[0]; temp2_p = X.PAA_pred.y[0]
            else:
                temp_p = X.PAA_pred.y[k - 3]; temp1_p = X.PAA_pred.y[k - 4]; temp2_p = X.PAA_pred.y[k - 5]

            if k == 1:
                Fpaa = pid_simple3(X.Fpaa.y[0], PAA_err, PAA_err1,
                                   temp_p, temp1_p, temp2_p,
                                   0, 150, 0.1, 0.50, 0 * 0.002, h)
            else:
                Fpaa = pid_simple3(X.Fpaa.y[k - 2], PAA_err, PAA_err1,
                                   temp_p, temp1_p, temp2_p,
                                   0, 150, 0.1, 0.50, 0 * 0.002, h)

    # =====================================================================
    #  Optional PAA set-point PI control (re-implementation of the APC of Goldrick et al., 2019)
    #  By default it reads the true PAA (X.PAA.y), as the released code does (idealised measurement).
    #  Set-point and start time are configurable; gains equal the built-in Raman controller (Kp 0.1, Ti 0.5 h, Td 0).
    # =====================================================================
    if getattr(Ctrl_flags, "PAA_setpoint_ctrl", False):
        PAA_sp = float(getattr(Ctrl_flags, "PAA_sp", 1250.0))
        start_h = float(getattr(Ctrl_flags, "PAA_ctrl_start_h", 25.0))
        if (k * h) >= start_h:                                # open loop (recipe Fpaa) before the start time
            se = getattr(Ctrl_flags, "paa_sensor_err", None)  # optional measurement-error series
            _efn = getattr(Ctrl_flags, "PAA_est_fn", None)
            if _efn is not None:                              # optional user-supplied PAA estimator
                pv0, pv1 = _efn(X, k, h)
                pv2 = pv1
            elif getattr(Ctrl_flags, "PAA_ctrl_real_pls", False) and getattr(X, "PAA_pred", None) is not None:
                # Optional: feed back the prediction of a recalibrated PLS model.
                # Units: X.PAA_pred is in g/L whereas X.PAA and the set-point are in mg/L,
                #   so the prediction is rescaled by PAA_pred_scale; without it the error stays at
                #   +set-point and Fpaa runs away. The released code uses PAA_pred only in the
                #   derivative term (gain 0) and never meets this mismatch.
                _sc = float(getattr(Ctrl_flags, "PAA_pred_scale", 1000.0))
                src = X.PAA_pred.y
                pv0 = (src[k - 3] if k > 3 else src[0]) * _sc
                pv1 = (src[k - 4] if k > 4 else src[0]) * _sc
                pv2 = (src[k - 5] if k > 5 else src[0]) * _sc
            elif se is not None:                              # measured PAA = true x (1 + error)
                pv0 = X.PAA.y[k - 2] * se[k - 2] if k > 2 else X.PAA.y[0] * se[0]
                pv1 = X.PAA.y[k - 3] * se[k - 3] if k > 3 else X.PAA.y[0] * se[0]
                pv2 = X.PAA.y[k - 4] * se[k - 4] if k > 4 else X.PAA.y[0] * se[0]
            else:                                            # idealised: true PAA
                pv0 = X.PAA.y[k - 2] if k > 2 else X.PAA.y[0]
                pv1 = X.PAA.y[k - 3] if k > 3 else X.PAA.y[0]
                pv2 = X.PAA.y[k - 4] if k > 4 else X.PAA.y[0]
            prev = X.Fpaa.y[0] if k == 1 else X.Fpaa.y[k - 2]
            _sp2 = getattr(Ctrl_flags, "PAA_sp2", None)       # optional second set-point (single set-point if unset)
            if _sp2 is not None and (k * h) >= getattr(Ctrl_flags, "PAA_t_sw", 1e9):
                PAA_sp = float(_sp2)
            Fpaa = pid_simple3(prev, PAA_sp - pv0, PAA_sp - pv1,
                               pv0, pv1, pv2, 0, float(getattr(Ctrl_flags, "PAA_umax", 150.0)),
                               0.1 * float(getattr(Ctrl_flags, "PAA_kp_mult", 1.0)), 0.50, 0.0, h)

    # =====================================================================
    #  Controller output vector
    # =====================================================================
    u["Fg"] = Fg
    u["RPM"] = 100
    u["Fs"] = Fs
    u["Fa"] = Fa
    u["Fb"] = Fb
    u["Fc"] = Fc
    u["Fh"] = Fh
    u["d1"] = ph_on_off
    u["tfl"] = temp_on_off
    u["Fw"] = Fw
    u["pressure"] = pressure
    u["viscosity"] = viscosity
    u["Fremoved"] = F_discharge
    u["Fpaa"] = Fpaa
    u["Foil"] = Foil
    u["NH3_shots"] = _get_channel_value(Xd, "NH3_shots", k, 0.0)

    if "Fault_ref" not in u:
        u["Fault_ref"] = 0

    # ---- optional multipliers on the operating variables ----
    # Ctrl_flags.lever = {name: multiplier (scalar or per-step array)}, set by an optimiser or controller.
    # pH_sp and T_sp are set-points handled above and may also vary with time.
    lever = getattr(Ctrl_flags, "lever", None)
    if lever:
        for _name in ("Fg", "RPM", "Fs", "Fw", "pressure", "Fremoved", "Fpaa", "Foil"):
            m = lever.get(_name)
            if m is None:
                continue
            if hasattr(m, "__len__"):
                m = m[min(k - 1, len(m) - 1)]
            u[_name] = u[_name] * m

    return u, X


# --------------------------------------------------------------------------
# Small helpers for the optional Xd channels (NH3_shots) used above.
# --------------------------------------------------------------------------
def _ensure_channel_value(Xd, name, N, k, value):
    """Set ``Xd.<name>.y[k-1] = value``, creating the channel if needed."""
    from .create_channel import create_channel
    ch = getattr(Xd, name, None)
    if ch is None or getattr(ch, "y", None) is None:
        ch = create_channel(name, "-", "h", np.zeros(N), np.zeros(N))
        setattr(Xd, name, ch)
    ch.y[k - 1] = value


def _get_channel_value(Xd, name, k, default):
    ch = getattr(Xd, name, None)
    if ch is None or getattr(ch, "y", None) is None:
        return default
    return ch.y[k - 1]
