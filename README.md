# Privacy-Preserving Quasi-Nonlinear Fuzzy Cognitive Maps

**Reproducibility package**

---

## Requirements

- Python >= 3.10
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- tqdm

## Layout

| Path | Contents |
|---|---|
| `code/` | Experiments, figure builder, table builder |
| `results/` | CSV files written by the experiments |

## Run order

```bash
cd code
python verify_claims.py                     # seconds
python exp1_main_sweep.py                   # ~20 min  -> exp1_main.csv
python exp2_sensitivity.py                  # ~15 min  -> exp2_talpha.csv
                                              #             exp2_phi.csv
                                              #             exp2_endpoints.csv
python exp3_overhead_baselines_attack.py    # ~20 min  -> exp3_overhead.csv
                                              #             exp4_baselines.csv
                                              #             exp5_attack.csv
python exp6_case_study.py                    # ~1 min   -> exp6_case_study.csv
                                              #             exp6_scenarios.csv
                                              #             lissos_Wp.npy
python make_figures.py                       # -> figs/*.pdf
python make_tables.py                        # -> results/tables.tex
                                              #    results/text_numbers.txt
```

Each script writes its own CSV files and can be rerun independently. Timings are quoted for a single core.

### Splitting long sweeps

Long sweeps can be split across runs.

- `exp1_main_sweep.py` takes a model range on the command line, for example:

  ```bash
  python exp1_main_sweep.py 0 9
  python exp1_main_sweep.py 9 18
  ```

- `exp2_sensitivity.py` and `exp3_overhead_baselines_attack.py` take the same range through the environment variables `M0` and `M1`, with a first argument selecting one part of the script, for example:

  ```bash
  M0=0 M1=9 python exp2_sensitivity.py talpha
  ```

Without arguments, each script runs its whole design and overwrites its CSV files.

## What produces what

| Output | Script | Source file |
|---|---|---|
| Table I | `make_tables.py` | `exp1_main.csv` |
| Table II | `make_tables.py` | `exp5_attack.csv` |
| Table III | `make_tables.py` | `exp3_overhead.csv` |
| Table IV | `make_tables.py` | `exp6_case_study.csv` |
| Section V-I | — | `exp6_scenarios.csv` |
| Section IV-D | `verify_claims.py` | `verify_claims.csv` |
| Figure 1 | `make_figures.py` | `exp1_main.csv` |
| Figure 2 | `make_figures.py` | `exp1_main.csv` |
| Figure 3 | `make_figures.py` | `exp4_baselines.csv` |
| Figure 4 | `make_figures.py` | `exp2_talpha.csv`, `exp2_phi.csv` |

Section IV-D, Proposition 1 and the transcription check of Section V-I are covered by `verify_claims.py`, which writes `results/verify_claims.csv`.

## Settings

- 18 synthetic models, with the number of concepts assigned systematically over the range 5 to 20, so model size is not confounded with model identity.
- Each model is driven by 100 initial conditions.
- Number of qFCM iterations (reasoning steps): `T = 20`
- Nonlinearity coefficient: `phi = 0.8` outside Section V-D
- L-BFGS-B budget: 1,000 iterations
- Seeds are fixed, so a rerun reproduces every CSV to the stored precision.

The scalability study of Table III is the one exception on model size, since it deliberately extends to 100 concepts.

## Notes on two implementation points

**Optimizer initialization.** The optimizer starts from a random stream independent of the one generating `W`. A start placed exactly at `W` would sit on a point where both gradient terms vanish, which would make the run vacuous.

**`reconstruction_attack` argument.** This function takes the model whose reasoning trace the adversary observes as an explicit argument.

- Passing `W'` reproduces the threat model of Section IV-A, in which the released model is the artifact that runs.
- Passing `W` models a deployment that publishes the original trajectories.

Section V-G reports both.
