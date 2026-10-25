# Reproducible ML Pipeline · Deliverable 05

Research Methods and Scientific Integrity in AI and Advanced Technologies · UNMSM · 2026-II
Student: Deisy Lizbeth Acosta Ticse · Instructor: Dr. Loveleen Gaur

**EN:** Logistic regression on the Wisconsin breast cancer dataset (569 rows), built from the course workshop notebook (NB4) so that a stranger can clone the repository and obtain exactly the same numbers.
**ES:** Regresión logística sobre el conjunto de datos de cáncer de mama de Wisconsin, construida a partir del notebook del taller (NB4) para que un extraño clone el repositorio y obtenga exactamente los mismos números.

## How to reproduce · Cómo reproducir
Tested with **Python 3.13.17** on Linux.
```bash
git clone <this-repo-url> repro-lab && cd repro-lab
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PYTHONHASHSEED=0 python -m src.train --seed 42          # one run
PYTHONHASHSEED=0 python -m src.run_seeds                # 5 seeds logged in MLflow
```
`PYTHONHASHSEED=0` must precede `python`: it only takes effect when the interpreter starts.

## Expected output · Resultado esperado
One run (`--seed 42`):
```
{"seed": 42, "data_md5": "5afc23b9622f8f9ffa60824f4e97d0d5", "accuracy_mean": 0.97367, "accuracy_sd": 0.01859}
```
Five seeds (`results_summary.csv`):

| seed | accuracy_mean | accuracy_sd (5 folds) |
|---|---|---|
| 0 | 0.97892 | 0.01593 |
| 1 | 0.97893 | 0.01466 |
| 2 | 0.97893 | 0.01466 |
| 3 | 0.98243 | 0.00620 |
| 42 | 0.97367 | 0.01859 |

**Reported result: accuracy 0.9786 ± 0.0031 (mean ± sample SD across 5 seeds, ddof = 1).**

## What makes it reproducible · Qué lo hace reproducible
| Requirement | How it is met |
|---|---|
| ≥ 4 commits | Project history in small, labelled steps (`git log`) |
| No data leakage | `StandardScaler` is inside a scikit-learn `Pipeline`, so in each cross-validation fold it is fit on the training folds only ([scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)) |
| Fixed randomness | Python and NumPy seeds set; the CV splitter receives the seed as an integer; `n_jobs=1` |
| 5 seeds in MLflow, mean ± SD | `src/run_seeds.py` logs seed, git commit, data MD5, model and metrics for each run (`mlflow.db`, kept out of Git); the summary is committed in `results_summary.csv` |
| Same data for everyone | Dataset committed in `data/` with its MD5 fingerprint recorded in every run |
| Exact environment | `requirements.txt` pins every imported library; Python version stated above |

To browse the runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Stranger test · Prueba del extraño
Before submission the repository was cloned into an empty folder, a new virtual environment was created from `requirements.txt` only, and both commands were run. The outputs matched the values above exactly (see commit history).

## Limitations · Limitaciones
Results are reproducible to the printed decimals on the stated Python/library versions; other operating systems or BLAS builds may differ in the last decimal. The SD across seeds measures variability due to the data split, not uncertainty about performance on new populations.

## AI-use disclosure
The pipeline follows the course notebook NB4. Claude (Anthropic) helped adapt it into scripts, write this README and run the stranger test; the author reviewed and ran the code.
