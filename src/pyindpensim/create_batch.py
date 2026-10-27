# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""create_batch.py

Python port of ``createBatch.m``.

Creates a *batch* structure with a given time duration. A batch is a
collection of :class:`~create_channel.Channel` objects; each channel holds the
time history for a state or manipulated variable.

The MATLAB code returned a struct ``X`` whose fields were channels. Here we
return a :class:`Batch` object. Channels are stored as attributes *and* kept in
an insertion-ordered dict so that the equivalent of MATLAB ``fieldnames(X)``
and ``rmfield`` is available (needed by ``generate_batch_records.py``).

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .create_channel import Channel, create_channel


class Batch:
    """Ordered collection of channels (equivalent to the MATLAB struct ``X``).

    Channels are reachable both as attributes (``X.S``) and through the
    ``channels`` ordered dict (``X.channels['S']``). Extra scalar/array fields
    used inside the simulation (e.g. ``PRBS_noise_addition``, ``Raman_Spec``,
    ``Stats``) can be attached as plain attributes as well.
    """

    def __init__(self):
        # Preserves insertion order -> mirrors MATLAB fieldnames() ordering.
        self.channels: "OrderedDict[str, Channel]" = OrderedDict()

    # -- attribute access that also registers Channel objects in the dict --
    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if isinstance(value, Channel):
            # 'channels' itself is set in __init__ before any Channel.
            self.__dict__.setdefault("channels", OrderedDict())
            self.channels[key] = value

    def fieldnames(self):
        """Equivalent of MATLAB ``fieldnames`` restricted to channels."""
        return list(self.channels.keys())

    def isfield(self, name):
        """Equivalent of MATLAB ``isfield``."""
        return name in self.channels or name in self.__dict__


def create_batch(h, T) -> Batch:
    """Build an empty batch structure.

    Parameters
    ----------
    h : float
        Sample time (hours).
    T : float
        Batch duration (hours). ``T/h`` must be (effectively) an integer.
    """
    N = int(round(T / h))
    t = np.zeros(N)
    y = np.zeros(N)

    X = Batch()

    # pensim manipulated variables
    X.Fg = create_channel('Aeration rate', 'L/h', 'h', t.copy(), y.copy())
    X.RPM = create_channel('Agitator RPM', 'RPM', 'h', t.copy(), y.copy())
    X.Fs = create_channel('Sugar feed rate', 'L/h', 'h', t.copy(), y.copy())
    X.sc = create_channel('Substrate feed concentration', 'g/L', 'h', t.copy(), y.copy())
    X.abc = create_channel('Acid/base feed concentration', 'moles', 'h', t.copy(), y.copy())
    X.Fa = create_channel('Acid flow rate', 'L/h', 'h', t.copy(), y.copy())
    X.Fb = create_channel('Base flow rate', 'L/h', 'h', t.copy(), y.copy())
    X.Fc = create_channel('Heating/cooling water flow rate', 'L/h', 'h', t.copy(), y.copy())
    X.Fh = create_channel('Heating water flow rate', 'L/h', 'h', t.copy(), y.copy())

    # indpensim manipulated variables
    X.Fw = create_channel('Water for injection/dilution', 'L/h', 'h', t.copy(), y.copy())
    X.pressure = create_channel('Air head pressure', 'bar', 'h', t.copy(), y.copy())
    X.Fremoved = create_channel('Dumped broth flow', 'L/h', 'h', t.copy(), y.copy())

    # pensim states
    X.S = create_channel('Substrate concentration', 'g/L', 'h', t.copy(), y.copy())
    X.DO2 = create_channel('Dissolved oxygen concentration', 'mg/L', 'h', t.copy(), y.copy())
    X.X = create_channel('Biomass concentration', 'g/L', 'h', t.copy(), y.copy())
    X.P = create_channel('Penicillin concentration', 'g/L', 'h', t.copy(), y.copy())
    X.V = create_channel('Vessel Volume', 'L', 'h', t.copy(), y.copy())
    X.Wt = create_channel('Vessel Weight', 'Kg', 'h', t.copy(), y.copy())
    X.pH = create_channel('pH', 'pH', 'h', t.copy(), y.copy())
    X.T = create_channel('Temperature', 'K', 'h', t.copy(), y.copy())
    X.Q = create_channel('Generated heat', 'kJ', 'h', t.copy(), y.copy())

    # indpensim states
    X.a0 = create_channel('type a0 biomass concentration', 'g/L', 'h', t.copy(), y.copy())
    X.a1 = create_channel('type a1 biomass concentration', 'g/L', 'h', t.copy(), y.copy())
    X.a3 = create_channel('type a3 biomass concentration', 'g/L', 'h', t.copy(), y.copy())
    X.a4 = create_channel('type a4 biomass concentration', 'g/L', 'h', t.copy(), y.copy())
    X.n0 = create_channel('state n0', '-', 'h', t.copy(), y.copy())
    X.n1 = create_channel('state n1', '-', 'h', t.copy(), y.copy())
    X.n2 = create_channel('state n2', '-', 'h', t.copy(), y.copy())
    X.n3 = create_channel('state n3', '-', 'h', t.copy(), y.copy())
    X.n4 = create_channel('state n4', '-', 'h', t.copy(), y.copy())
    X.n5 = create_channel('state n5', '-', 'h', t.copy(), y.copy())
    X.n6 = create_channel('state n6', '-', 'h', t.copy(), y.copy())
    X.n7 = create_channel('state n7', '-', 'h', t.copy(), y.copy())
    X.n8 = create_channel('state n8', '-', 'h', t.copy(), y.copy())
    X.n9 = create_channel('state n9', '-', 'h', t.copy(), y.copy())
    X.nm = create_channel('state nm', '-', 'h', t.copy(), y.copy())
    X.phi0 = create_channel('state phi0', '-', 'h', t.copy(), y.copy())
    X.CO2outgas = create_channel('carbon dioxide percent in off-gas', '%', 'h', t.copy(), y.copy())
    X.Culture_age = create_channel('Cell culture age', 'h', 'h', t.copy(), y.copy())
    X.Fpaa = create_channel('PAA flow', 'PAA flow (L/h)', 'h', t.copy(), y.copy())
    X.PAA = create_channel('PAA concentration', 'PAA (g L^{-1})', 'h', t.copy(), y.copy())
    X.PAA_offline = create_channel('PAA concentration offline', 'PAA (g L^{-1})', 'h', t.copy(), y.copy())
    X.Foil = create_channel('Oil flow', 'L/hr', 'h', t.copy(), y.copy())
    X.NH3 = create_channel('NH_3 concentration', 'NH3 (g L^{-1})', 'h', t.copy(), y.copy())
    X.NH3_offline = create_channel('NH_3 concentration off-line', 'NH3 (g L^{-1})', 'h', t.copy(), y.copy())
    X.OUR = create_channel('OUR', 'OUR (g min^{-1})', 'h', t.copy(), y.copy())
    X.O2 = create_channel('Oxygen in percent in off-gas', 'O2  (%)', 'h', t.copy(), y.copy())
    X.mup = create_channel('Specific growth rate of Penicillin', 'mu_P (h^{-1})', 'h', t.copy(), y.copy())
    X.mux = create_channel('Specific growth rate of Biomass', 'mu_X (h^{-1})', 'h', t.copy(), y.copy())
    X.P_offline = create_channel('Offline Penicillin concentration', 'P(g L^{-1})', 'h', t.copy(), y.copy())
    X.X_CER = create_channel('Biomass concentration from CER', 'g min^{-1}', 'h', t.copy(), y.copy())
    X.X_offline = create_channel('Offline Biomass concentratio', 'X(g L^{-1})', 'h', t.copy(), y.copy())
    X.CER = create_channel('Carbon evolution rate', 'g/h', 'h', t.copy(), y.copy())
    X.mu_X_calc = create_channel('Biomass specific growth rate ', 'hr^{-1}', 'h', t.copy(), y.copy())
    X.mu_P_calc = create_channel('Penicillin specific growth rate ', 'hr^{-1}', 'h', t.copy(), y.copy())
    X.F_discharge_cal = create_channel('Discharge rate', 'L hr^{-1}', 'h', t.copy(), y.copy())
    X.NH3_shots = create_channel('Ammonia shots', 'kgs', 'h', t.copy(), y.copy())
    # NB: the MATLAB source re-declares OUR here; kept identical for fidelity.
    X.OUR = create_channel('Oxygen Uptake Rate ', '(g min^{-1})', 'h', t.copy(), y.copy())
    X.CO2_d = create_channel('Dissolved CO_2  ', '(mg L^{-1})', 'h', t.copy(), y.copy())
    X.Viscosity = create_channel('Viscosity ', 'centPoise', 'h', t.copy(), y.copy())
    X.Viscosity_offline = create_channel('Viscosity', 'centPoise', 'h', t.copy(), y.copy())
    X.Fault_ref = create_channel('Fault reference', 'Fault ref', 'h', t.copy(), y.copy())
    X.Control_ref = create_channel('0 - Recipe driven 1 - Operator controlled', 'Control ref', 'Batch number', t.copy(), y.copy())
    X.PAT_ref = create_channel('1- No Raman spec, 1-Raman spec recorded,2-PAT control', 'PAT ref', 'Batch number', t.copy(), y.copy())
    X.Batch_ref = create_channel('Batch reference', 'Batch ref', 'Batch ref', t.copy(), y.copy())

    return X
