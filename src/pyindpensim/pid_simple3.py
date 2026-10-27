# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""pid_simple3.py

Python port of ``PIDSimple3.m``.

Simple PID controller in incremental form. The derivative component is applied
only to the output (PV). The controller also checks for control-signal
saturation. Integral and derivative time constants must be expressed in the
same time unit as the sampling period.

    u_k = u_{k-1} + Kp * [ (e_k - e_{k-1}) + e_k*h/Ti + (-Td/h)(y_k - 2y_{k-1} + y_{k-2}) ]

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations


def pid_simple3(uk1, ek, ek1, yk, yk1, yk2, u_min, u_max, Kp, Ti, Td, h):
    """Incremental PID with output-derivative and saturation.

    Parameters
    ----------
    uk1 : float
        Previous control input.
    ek, ek1 : float
        Current and previous error.
    yk, yk1, yk2 : float
        Current and two previous process values (PV).
    u_min, u_max : float
        Saturation limits on the control signal.
    Kp : float
        Proportional gain.
    Ti : float
        Integral time constant.
    Td : float
        Derivative time constant.
    h : float
        Sampling time.

    Returns
    -------
    float
        The (saturated) control signal ``u``.
    """
    # proportional component
    P = ek - ek1

    # integral component (only if the integral time constant is defined)
    if Ti > 1e-7:
        I = ek * h / Ti
    else:
        I = 0.0

    # derivative component (applied to the output)
    if Td > 0.001:
        D = -Td / h * (yk - 2 * yk1 + yk2)
    else:
        D = 0.0

    # compute and saturate the control signal
    u = uk1 + Kp * (P + I + D)
    if u > u_max:
        u = u_max
    if u < u_min:
        u = u_min

    return u
