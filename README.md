# pyIndPenSim

A Python port of **IndPenSim**, the industrial-scale fed-batch penicillin fermentation simulator of
Goldrick et al. (2015, 2019). IndPenSim models the production of penicillin by *Penicillium chrysogenum*
in a 100,000 L fed-batch fermenter with 33 ordinary differential equations, and includes a simulated
Raman probe and a PLS model that predicts the phenylacetic acid (PAA) concentration from its spectra.

Each module of this package mirrors one `.m` file of IndPenSim V2.02. The port was written for the
study *Process optimisation of fed-batch fermentation using online measurements alone* (Hong, submitted),
in which every simulated batch was generated with this code.

> pyIndPenSim is an independent port. It is not affiliated with, or endorsed by, the authors of
> IndPenSim or their institutions.

## Installation

```bash
git clone https://github.com/sehongkopo/pyIndPenSim.git
cd pyIndPenSim
pip install -e .                 # numpy, scipy
pip install -e ".[validate]"     # adds pandas for examples/validate_table1.py
```

Python 3.9 or later is required.

## Quick start

```python
import numpy as np
from pyindpensim import indpensim_run, batch_flags, net_yield_kg

X = indpensim_run(1, batch_flags(raman=0), rng=np.random.default_rng(2000), rtol=1e-6, atol=1e-8)
print(net_yield_kg(X))           # 2334.68 kg for seed 2000
print(X.PAA.t[:5], X.PAA.y[:5])  # every recorded variable is a channel with .t and .y
```

`batch_flags()` builds the run flags of the released code:

| Argument | Values |
|---|---|
| `control_strategy` | 0 recipe-driven, 1 operator-like PRBS perturbations of the feeds (default) |
| `variable_length` | 0 fixed batch length, 1 uneven batch length (default) |
| `fault` | fault reference of the released code, 0 none (default) |
| `raman` | 0 no Raman (default), 1 record Raman spectra, 2 Raman-based PAA control |

A batch is fully determined by the `numpy.random.Generator` passed as `rng`, so the same seed gives the
same batch.

`examples/run_batch.py` runs one batch with Raman-based control off and on.

## Differences from the MATLAB code

- **Solver.** `ode15s` is replaced by the BDF method of `scipy.integrate.solve_ivp`. The study used
  `rtol = 1e-6`, `atol = 1e-8` and `solv = 2`; the defaults of `indpensim_run` (`1e-3`, `1e-6`) follow
  the released code and are faster but less accurate.
- **Random numbers.** The MATLAB random stream cannot be reproduced in Python, so no pyIndPenSim batch
  is a copy of a released batch. Agreement is therefore assessed on distributions (see Validation).
- **Raman spectra.** With `raman = 1` or `2` the spectrum simulation draws about 2,200 random numbers per
  step from the same generator, so the same seed gives a different batch with Raman on and off.
- **Raman-based control uses the true PAA.** As in the released code, the error of the Raman PI
  controller is computed from the true PAA concentration; the PLS prediction enters only the
  derivative term, whose gain is zero.

## Extensions

The following optional inputs are not part of the released code. When they are not set, the
simulation is identical to the port of the released code.

| Input | Where | Effect |
|---|---|---|
| `lever` | `indpensim_run(..., lever={...})` | multipliers on the recipe values of `Fs`, `Fpaa`, `Foil`, `Fg`, `Fw`, `Fremoved`, `RPM`, `pressure`, as a scalar or one value per 0.2 h step |
| `pH_sp`, `T_sp` | `indpensim_run(...)` | pH and temperature set-points, scalar or per step |
| `PAA_setpoint_ctrl=True` with `PAA_sp`, `PAA_ctrl_start_h` | `batch_flags(...)` | PAA set-point PI control on the true PAA without simulating spectra (the APC of Goldrick et al., 2019, with `PAA_sp=1200.0`, `PAA_ctrl_start_h=10.0`) |
| `PAA_est_fn`, `PAA_sp2`, `PAA_t_sw`, `PAA_kp_mult`, `PAA_umax` | `batch_flags(...)` | the same PI on a user-supplied PAA estimate, with an optional second set-point, gain multiplier and output limit |
| `PAA_ctrl_use_pred` with `PAA_sensor_sigma`, `PAA_sensor_bias`, `PAA_sensor_model` | `batch_flags(...)` | the same PI on a PAA reading with a simulated relative error |
| `_rec={}` | `indpensim_run(...)` | records the state `y`, input `u` and parameters `par` at every step |
| `MU_SIGMA_MULT` | environment variable | scales the batch-to-batch spread of the two strain growth parameters (1.0 = released) |

The data files of the Raman model (`PAA_PLS_model.mat`, `reference_Specra.txt`) are shipped in
`src/pyindpensim/data`. To use other copies, set the environment variable `PYINDPENSIM_DATA_DIR`
or pass `data_dir` to the loaders in `raman_sim` and `substrate_prediction`.

## Validation

`examples/validate_table1.py` reproduces the batch-yield rows of Table 1 of the accompanying study.
It needs the per-batch summary of the released 100-batch dataset of Goldrick et al. (2019),
`100_Batches_IndPenSim_Statistics.csv`, which is distributed with that dataset and is not included here.

```bash
python examples/validate_table1.py --stats /path/to/100_Batches_IndPenSim_Statistics.csv --n 100 --workers 8
```

| Raman-based control | Released (batches) | pyIndPenSim (seeds 2000–2099) | Difference | Cohen's d | KS p | MWU p |
|---|---|---|---|---|---|---|
| Off | 2,913 ± 768 kg (1–30) | 2,894 ± 802 kg | −18 kg (−0.63 %) | −0.023 | 0.856 | 0.836 |
| On | 3,484 ± 308 kg (61–90) | 3,481 ± 487 kg | −3 kg (−0.08 %) | −0.006 | 0.278 | 0.709 |

`tests/test_smoke.py` checks that seed 2000 reproduces the yields recorded in the study
(`pip install -e ".[test]"`, then `pytest`).

## Licence

BSD 3-Clause. The MATLAB original is © 2016 Stephen Goldrick and is distributed under the BSD 3-Clause
licence reproduced verbatim in [`LICENSE-IndPenSim`](LICENSE-IndPenSim); the port and its extensions are
© 2026 Seong-Eui Hong under the same terms ([`LICENSE`](LICENSE)).

## Citation

Please cite this software (see [`CITATION.cff`](CITATION.cff)) together with the original papers:

- Goldrick, S., Ştefan, A., Lovett, D., Montague, G., Lennox, B., 2015. The development of an
  industrial-scale fed-batch fermentation simulation. *J. Biotechnol.* 193, 70–82.
  https://doi.org/10.1016/j.jbiotec.2014.10.029
- Goldrick, S., Duran-Villalobos, C.A., Jankauskas, K., Lovett, D., Farid, S.S., Lennox, B., 2019.
  Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation
  process. *Comput. Chem. Eng.* 130, 106471. https://doi.org/10.1016/j.compchemeng.2019.05.037
