# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""substrate_prediction.py

Python port of ``Substrate_prediction.m``.

Predicts the phenylacetic acid (PAA) concentration from the simulated Raman
spectrum using a pre-fitted PLS model (``PAA_PLS_model.mat``, which stores the
regression-coefficient matrix ``b``). The prediction is written into the
``PAA_pred`` channel and used by the Raman-based PAA controller
(``fctrl_indpensim`` when ``Raman_spec == 2``).

MATLAB specifics reproduced here
--------------------------------
* ``j = k - 1``  (prediction lags the current sample by one).
* ``sgolayfilt(x, 2, 5)``  ->  Savitzky-Golay, polyorder 2, window 5.
* peak selection ``d([350:500 800:860])`` (1-based, inclusive) ->
  0-based ``np.r_[349:500, 799:860]``.
* ``No_LV = 4`` latent variables  ->  row ``b[3, :]`` (0-based).

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter

from .create_channel import create_channel

def _default_data_dir():
    """Directory holding the IndPenSim data files.

    Order: the ``PYINDPENSIM_DATA_DIR`` environment variable, then the ``data``
    folder shipped with this package.
    """
    return os.environ.get("PYINDPENSIM_DATA_DIR") or os.path.join(os.path.dirname(__file__), "data")

_PLS_COEFF_CACHE = {}


def _load_pls_coeff(data_dir=None):
    """Load and cache the PLS regression-coefficient matrix ``b``."""
    data_dir = data_dir or _default_data_dir()
    path = os.path.join(data_dir, "PAA_PLS_model.mat")
    if path not in _PLS_COEFF_CACHE:
        mat = loadmat(path)
        if "b" not in mat:
            raise KeyError(
                f"'b' (PLS coefficients) not found in {path}; keys={list(mat)}"
            )
        _PLS_COEFF_CACHE[path] = np.asarray(mat["b"], dtype=float)
    return _PLS_COEFF_CACHE[path]


# 0-based equivalent of MATLAB [350:500 800:860]
_PAA_PEAK_IDX = np.r_[349:500, 799:860]
_NO_LV = 4  # number of latent variables (MATLAB 1-based row index)

# Optional recalibrated PLS: when set, this (coef, intercept) replaces PAA_PLS_model.mat as the soft sensor.
_OVERRIDE_COEF = None
_OVERRIDE_INTERCEPT = 0.0
def set_pls_override(coef, intercept=0.0):
    global _OVERRIDE_COEF, _OVERRIDE_INTERCEPT
    _OVERRIDE_COEF = None if coef is None else np.asarray(coef, float)
    _OVERRIDE_INTERCEPT = float(intercept)


def substrate_prediction(k, X, h, T, data_dir=None):
    """Predict PAA from the Raman spectrum for sample ``k``.

    Parameters
    ----------
    k : int
        Current 1-based sample index (matches the MATLAB loop counter).
    X : Batch
        Batch structure holding ``Raman_Spec`` and ``PAA_pred``.
    h, T : float
        Sampling period and batch length (unused numerically, kept for parity).
    data_dir : str, optional
        Directory containing ``PAA_PLS_model.mat``.

    Returns
    -------
    Batch
        The (mutated) batch ``X``.
    """
    b = _load_pls_coeff(data_dir)
    j = k - 1  # MATLAB: j = k-1 (1-based sample number)

    # Ensure the PAA_pred channel exists and is long enough.
    N = X.Raman_Spec.Intensity.shape[1]
    if not X.isfield("PAA_pred") or getattr(X, "PAA_pred", None) is None:
        X.PAA_pred = create_channel("PAA prediction", "PAA (mg L^{-1})", "h",
                                    np.zeros(N), np.zeros(N))

    # Column j (1-based) -> index j-1. Smooth, differentiate, pick peaks.
    spectrum = X.Raman_Spec.Intensity[:, j - 1]
    raman_sg = savgol_filter(spectrum, window_length=5, polyorder=2)
    raman_sg_d = np.diff(raman_sg)
    paa_peaks_spec = raman_sg_d[_PAA_PEAK_IDX]

    # PLS projection onto latent-variable row No_LV (1-based) -> b[No_LV-1].
    # Use the recalibrated PLS (coef, intercept) when it has been set.
    if _OVERRIDE_COEF is not None:
        pred = float(paa_peaks_spec @ _OVERRIDE_COEF + _OVERRIDE_INTERCEPT)
    else:
        pred = float(paa_peaks_spec @ b[_NO_LV - 1, :])
    X.PAA_pred.y[j - 1] = pred

    # 3-point moving average once enough history is available (MATLAB j > 20).
    if j > 20:
        X.PAA_pred.y[j - 1] = (
            X.PAA_pred.y[j - 2] + X.PAA_pred.y[j - 3] + X.PAA_pred.y[j - 1]
        ) / 3.0

    return X
