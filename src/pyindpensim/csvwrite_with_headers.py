# Ported from IndPenSim V2.02 (MATLAB) by Stephen Goldrick et al.
# Copyright (c) 2016, Stephen Goldrick. Python port copyright (c) 2026, Seong-Eui Hong.
# Distributed under the BSD 3-Clause License; see LICENSE and LICENSE-IndPenSim.
"""csvwrite_with_headers.py

Python port of ``csvwrite_with_headers.m``.

Writes a numeric 2-D array to a CSV file with a leading header row. Mirrors the
MATLAB helper (which wrapped ``csvwrite``/``dlmwrite``) including the optional
row/column offsets.

Original author: Stephen Goldrick.
Reference: DOI 10.1016/j.jbiotec.2014.10.029 and
https://doi.org/10.1016/j.compchemeng.2019.05.037
"""

from __future__ import annotations

import numpy as np


def csvwrite_with_headers(filename, m, headers, r=0, c=0):
    """Write ``m`` to ``filename`` as CSV with a header row.

    Parameters
    ----------
    filename : str
        Output filename.
    m : array_like
        2-D numeric array of data.
    headers : sequence of str
        Column headers; length must equal the number of columns in ``m``.
    r : int, optional
        Row offset of the data (blank leading rows / header comma shift).
    c : int, optional
        Column offset of the data (blank leading columns).
    """
    if not isinstance(filename, str):
        raise TypeError("FILENAME must be a string")

    m = np.atleast_2d(np.asarray(m, dtype=float))
    headers = [str(hh) for hh in headers]

    if len(headers) != m.shape[1]:
        raise ValueError(
            "number of header entries must match the number of columns in the data")

    # Build the header string (comma-separated), with optional right shift.
    header_string = ",".join(headers)
    if r > 0:
        header_string = "," * r + header_string

    with open(filename, "w", newline="") as fid:
        # MATLAB used '\r\n' line terminators for the header.
        fid.write(header_string + "\r\n")

    # Append the data with the requested column offset.
    if c > 0:
        pad = np.zeros((m.shape[0], c))
        m_out = np.hstack([pad, m])
    else:
        m_out = m

    with open(filename, "a", newline="") as fid:
        for row in m_out:
            fid.write(",".join(repr(float(v)) for v in row) + "\n")
