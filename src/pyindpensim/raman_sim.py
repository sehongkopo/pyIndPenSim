# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""raman_sim.py

Python port of ``Raman_Sim.m``.

Generates a simulated Raman spectrum for the current sample. The spectrum is
built from a reference spectrum (``reference_Specra.txt``) plus:
  * a global intensity trend driven by product/biomass/viscosity/time,
  * an accumulated, smoothed random-walk noise term,
  * Gaussian "bell-curve" peaks for glucose (substrate), PAA and product,
    each scaled by the corresponding simulated concentration.

The result is stored in ``X.Raman_Spec.Intensity[:, k-1]`` (a 2200 x N array,
columns indexed by the 1-based sample ``k``).

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

import os

import numpy as np

_WAVENUMBER_MAX = 2200

def _default_data_dir():
    """Directory holding the IndPenSim data files.

    Order: the ``PYINDPENSIM_DATA_DIR`` environment variable, then the ``data``
    folder shipped with this package.
    """
    return os.environ.get("PYINDPENSIM_DATA_DIR") or os.path.join(os.path.dirname(__file__), "data")

_REF_SPECTRA_CACHE = {}


class RamanSpec:
    """Container mirroring the MATLAB ``X.Raman_Spec`` struct."""

    def __init__(self, n_samples, wavenumber_max=_WAVENUMBER_MAX):
        self.Wavelength = None
        self.Intensity = np.zeros((wavenumber_max, n_samples))


def _load_reference_spectra(data_dir=None):
    """Load and cache ``reference_Specra.txt`` (columns: wavelength, intensity)."""
    data_dir = data_dir or _default_data_dir()
    path = os.path.join(data_dir, "reference_Specra.txt")
    if path not in _REF_SPECTRA_CACHE:
        _REF_SPECTRA_CACHE[path] = np.loadtxt(path)
    return _REF_SPECTRA_CACHE[path]


def _matlab_smooth(x, span):
    """Replicate MATLAB ``smooth(x, span)`` (default 'moving' average).

    ``span`` is forced odd; interior points use a centred moving average, while
    the first/last points use progressively smaller symmetric windows.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if span % 2 == 0:
        span -= 1
    span = min(span, n if n % 2 == 1 else n - 1)
    half = (span - 1) // 2
    y = np.empty_like(x)
    for i in range(n):
        w = min(i, n - 1 - i, half)  # symmetric, shrinking window at edges
        y[i] = x[i - w:i + w + 1].mean()
    return y


def _add_gaussian_peak(target, center_1based, width, length_factor=2,
                       divisor=1.0):
    """Add a Gaussian peak into ``target`` (length 2200), MATLAB-style.

    center is 1-based; std_dev = width/2; the peak spans
    ``+/- width*length_factor`` around the centre.
    """
    length = width * length_factor
    std_dev = width / 2.0
    xs = np.arange(-length, length + 1)
    vals = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * (xs / std_dev) ** 2
    ) / divisor
    idx = xs + center_1based - 1  # 0-based target indices
    valid = (idx >= 0) & (idx < target.size)
    target[idx[valid]] += vals[valid]
    return target


def raman_sim(k, X, h, T, rng=None, data_dir=None):
    """Build the simulated Raman spectrum for sample ``k`` (1-based).

    Parameters
    ----------
    k : int
        Current 1-based sample index.
    X : Batch
        Batch structure. ``X.Raman_Spec`` is created on first call.
    h, T : float
        Sampling period and batch length (``N = T/h``).
    rng : numpy.random.Generator, optional
        Random source for the noise term. Defaults to a fresh default_rng().
    data_dir : str, optional
        Directory containing ``reference_Specra.txt``.
    """
    if rng is None:
        rng = np.random.default_rng()

    Wmax = _WAVENUMBER_MAX
    N = int(round(T / h))

    # Lazily create the Raman_Spec container.
    if not X.isfield("Raman_Spec") or getattr(X, "Raman_Spec", None) is None:
        X.Raman_Spec = RamanSpec(N, Wmax)

    # --- global intensity trend --------------------------------------------
    j_idx = np.arange(1, Wmax + 1)  # 1..2200 (1-based like MATLAB)
    intensity_shift1 = np.exp(j_idx / (Wmax * 0.5)) - 0.5

    a = -0.00178143846614472 * 0.1
    b = 1.05644816081515
    c = -0.0681439987249108 * 0.1
    d = -0.02
    product_s = X.P.y[k - 1] / 40.0           # penicillin scaled
    biomass_s = X.X.y[k - 1] / 40.0           # biomass scaled
    viscosity_s = X.Viscosity.y[k - 1] / 100.0
    time_s = k / (T / h)
    intensity_increase = a * biomass_s + b * product_s + c * viscosity_s + d * time_s

    scaling_factor = 370000.0
    gluc_increase = 800000.0 * 3 / 1400
    paa_increase = 1700000.0 / 1000
    prod_increase = 100000.0

    # --- reference spectrum -------------------------------------------------
    reference_spectra_2200 = _load_reference_spectra(data_dir)
    X.Raman_Spec.Wavelength = reference_spectra_2200[:Wmax, 0]
    reference_spectra = reference_spectra_2200[:Wmax, 1]

    new_spectra = intensity_increase * intensity_shift1 * scaling_factor + reference_spectra

    # --- accumulated random-walk noise -------------------------------------
    noise_factor = 50.0
    random_number = rng.integers(1, 4, size=Wmax)  # randi(3) -> {1,2,3}
    random_noise = np.where(random_number == 1, 0.0,
                            np.where(random_number == 2, noise_factor, -noise_factor))
    random_noise_summed = np.cumsum(random_noise)
    random_noise_summed_smooth = _matlab_smooth(random_noise_summed, 25)

    new_spectra_noise = new_spectra + 10.0 * random_noise_summed_smooth

    # --- Gaussian peaks -----------------------------------------------------
    glucose = np.zeros(Wmax)
    _add_gaussian_peak(glucose, 219, 70)                 # peak A
    _add_gaussian_peak(glucose, 639, 20, divisor=4.3)    # peak B
    _add_gaussian_peak(glucose, 1053, 100)               # peak C

    paa = np.zeros(Wmax)
    _add_gaussian_peak(paa, 419, 60)                     # peak A
    _add_gaussian_peak(paa, 839, 15, divisor=4.3)        # peak B

    product = np.zeros(Wmax)
    _add_gaussian_peak(product, 800, 30, length_factor=4)    # Pen G peak a
    _add_gaussian_peak(product, 1200, 30, length_factor=30)  # Pen G peak b

    K_G = 0.005
    substrate_raman = X.S.y[k - 1]
    paa_raman = X.PAA.y[k - 1]

    spectrum = (
        new_spectra_noise
        + glucose * gluc_increase * substrate_raman / (K_G + substrate_raman)
        + paa * paa_increase * paa_raman
        + product * prod_increase * X.P.y[k - 1]
    )

    X.Raman_Spec.Intensity[:, k - 1] = spectrum
    return X
