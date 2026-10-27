# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""create_channel.py

Python port of ``createChannel.m``.

Creates a *channel* structure. A channel holds the time history of a single
state / manipulated variable together with a description, its measurement
unit and its time unit.

MATLAB original members: {name, yUnit, tUnit, t, y}

Notes on the port
-----------------
* The MATLAB function was overloaded on ``nargin`` (3 args -> metadata only,
  5 args -> metadata + data). Here the data arguments ``t`` and ``y`` are
  simply optional. When supplied they are stored as 1-D ``numpy`` arrays
  (column vectors in MATLAB become plain 1-D arrays in Python).
* Indexing convention: MATLAB uses 1-based indexing (``c.y(k)``); the Python
  port stores ``t``/``y`` as 0-based ``numpy`` arrays (``c.y[k-1]``).

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Channel:
    """A single measurement/state channel (equivalent to a MATLAB struct)."""

    name: str
    yUnit: str
    tUnit: str
    t: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


def create_channel(name, measurement_unit, time_unit, t=None, y=None) -> Channel:
    """Create a :class:`Channel`.

    Parameters
    ----------
    name : str
        Channel name.
    measurement_unit : str
        Channel measurement unit (stored as ``yUnit``).
    time_unit : str
        Time unit (stored as ``tUnit``).
    t : array_like, optional
        Vector of time instances.
    y : array_like, optional
        Vector of measurements. Must have the same length as ``t``.
    """
    # --- validate the string metadata (mirrors the MATLAB ischar checks) ---
    if not isinstance(name, str):
        raise ValueError("Name is not a string!")
    if not isinstance(measurement_unit, str):
        raise ValueError("Measurement unit is not a string!")
    if not isinstance(time_unit, str):
        raise ValueError("Time unit is not a string!")

    # --- metadata-only case (MATLAB nargin == 3) ---
    if t is None and y is None:
        return Channel(name=name, yUnit=measurement_unit, tUnit=time_unit)

    # --- metadata + data case (MATLAB nargin == 5) ---
    if t is None or y is None:
        raise ValueError("Both t and y must be provided together.")

    t_arr = np.asarray(t, dtype=float).reshape(-1)  # force column -> 1-D
    y_arr = np.asarray(y, dtype=float).reshape(-1)

    if not (np.isrealobj(t_arr) and np.isrealobj(y_arr)):
        raise ValueError("t or y are not real arrays!")

    if t_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("t and y dont have the same number of rows!")

    return Channel(name=name, yUnit=measurement_unit, tUnit=time_unit,
                   t=t_arr, y=y_arr)
