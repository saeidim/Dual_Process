# Thinking Fast and Slow at the Network Edge — reproduction code

Budget-constrained metacognitive deliberation for world-model AoI scheduling at a
mmWave RSU.  Every number, table and figure in `../paper/` is produced here.

## Files
| file | content |
|---|---|
| `v2xsim.py` | the same physics-grounded simulator and world model as Paper 1 |
| `p2_core.py` | System 1 (max-weight index), System 2 (Monte-Carlo rollouts in the world model with common random numbers), the free risk signal, isotonic value map, and the gates: always-S1/S2, random, periodic, uncertainty, static threshold (Thm 4), virtual queue (Thm 5), oracle |
| `run_p2_experiments.py` | calibration, budget frontier, V trade-off with budget traces, oracle gate, rollout count and horizon studies -> `results/*.json` |
| `run_p2_vq.py` | second pass: virtual-queue runs with the warm-started queue |
| `run_p2_more.py` | final pass: virtual queue at V=10 and five more realizations (8 total) for every gate at rho=0.1 |
| `run_p2_expansion.py` | journal-version experiments: multi-planner calibration (reference-measured gains) and evaluation, dense scenario, design study with 6 realizations, tolerable-age sensitivity, world-model mismatch |
| `run_p2_dense30.py` | dense scenario with a 30% budget |
| `make_figures.py` | builds `../paper/figs/*.pdf` and `../paper/numbers.tex` |

| `run_revision.py`, `make_figures_rev.py` | first revision: all results of `results_rev/` and `numbers.tex` |
| `run_revision2.py` | second revision: tiny/cheap planners in every slot, gates with the tiny planner at small budgets, three-planner menu, cost-aware System 1, pipelined (delayed) deliberation, learned value map, biased world models, 24 000-slot traces -> `results_rev2/*.json` |
| `eps_diag2.py` | change of the imagined delivery probabilities under the biased world models (Lemma 1) |
| `make_numbers_rev2.py` | builds `numbers2.tex` and `figs/fig_compute.pdf` |

## Reproduce
```bash
pip install -r requirements.txt
python run_p2_experiments.py      # ~60 min on 2 cores (use --quick for a smoke test)
python run_p2_vq.py               # ~15 min
python run_p2_more.py             # ~25 min
python run_p2_expansion.py        # ~85 min
python run_p2_dense30.py          # ~8 min
python make_figures.py
python run_revision.py && python make_figures_rev.py        # first revision (~3.5 h on 2 cores)
python run_revision2.py && python eps_diag2.py && python make_numbers_rev2.py   # second revision (~4 h)
cd ../paper && latexmk -pdf main.tex
```

Note: `run_p2_experiments.py` also evaluates an 'estimated-gain' gate (`oracle`); it selects on noisy System 2 estimates and is not reported in the paper.
