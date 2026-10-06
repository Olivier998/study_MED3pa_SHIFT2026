# study_MED3pa_SHIFT2026

Code for the SMASH 2026 short paper *Diagnosing Low-Confidence Patient Profiles via Localized Harmful Covariate Shift Detection*.

MED3pa identifies low-confidence patient profiles of an MLP base model, and the
Detectron is applied within each profile to test whether its performance loss is
explained by a covariate shift. Shifts are induced in a controlled way by excluding
a subgroup from the training set only, for three MIMIC-IV tasks: 30-day
readmission (`30dr`), death within 48 hours (`mortality_48h`) and discharge alive
within 48 hours (`disch_alive_48h`).

| Criterion         | Training set restricted to | Tasks     |
|-------------------|----------------------------|-----------|
| `age_gr_50`       | patients older than 50     | all       |
| `sex_m_eq_1`      | male patients              | all       |
| `race_WHITE_eq_1` | White patients             | all       |

## Installation

Python 3.12:

```bash
pip install -r requirements.txt
```

MED3pa is installed from the `dev_explainer` branch of
[MEDomicsLab/MED3pa](https://github.com/MEDomicsLab/MED3pa) (pinned commit), which
provides the uncertainty explainer (Detectron, FeatureResumer).

## Data

The datasets are derived from [MIMIC-IV](https://physionet.org/content/mimiciv/)
and cannot be redistributed. Place one CSV per task in `datasets/`
(`30dr.csv`, `mortality_48h.csv`, `disch_alive_48h.csv`), with a binary `target`
column. Categorical columns (e.g. `gender`, `race`) are one-hot encoded by the
pipeline, which produces the `gender_M` and `race_WHITE` columns used by the
exclusion criteria.

## Usage

From the repository root:

```bash
# 1. Base models + MED3pa/Detectron results for every (task, criterion)
python -m experiments.main

# 2. Collect the profiles and their Detectron results from the profile trees
python -m experiments.extract_profiles

# 3. Base model metrics on every profile, for all / excluded / included patients
python -m experiments.main --profile-metrics experiments/results/profiles.csv
```

`--test` runs the pipeline on the first 200 rows of each dataset (outputs in
`experiments/results_test/`).

## Outputs

`experiments/results/{task}/{criterion}/` contains the fitted preprocessing objects
and base model (`*.pkl`, reused on later runs), the selected MLP hyperparameters,
the base model metrics, the MED3pa results, the MDR curves (`mdr.svg`), the
interactive profile tree (`profiles_*.html`) and the run metadata.

Step 2 writes `experiments/results/profiles.csv`. Step 3 writes one JSON per
(task, criterion) in `experiments/results/profile_metrics/`.

## Repository structure

```
experiments/
    main.py               experiment pipeline (configuration at the top)
    extract_profiles.py   profiles + Detectron results from the profile trees
src/
    mlp.py                PyTorch MLP base model and Optuna tuning
    preprocessing.py      cleaning, encoding, exclusion criteria, profile masks
    metrics.py            base model metrics
    profiles.py           parsing of the MED3pa profile trees
```
