# Partial Membership Inference Attacks

Research code for membership inference attacks (MIAs) against machine-learning classifiers when the attacker only has **partial or perturbed** knowledge of a record's features.

The attack follows the shadow-model approach: shadow classifiers imitate the target model, their posterior vectors on member (`IN`) and non-member (`OUT`) records train one binary attack model per class, and the attack models then query the target. The partial-membership experiments perturb the queried records first (noise, masking, feature selection, or enumeration of unknown binary features) and measure how well membership can still be inferred.

The accompanying report is [docs/PMIA.pdf](docs/PMIA.pdf).

## Repository layout

```
pmia/                  Corrected, tested implementation (use this)
  datastructures.py    Dataset wrappers, splitting/sampling, classifier collections
  mia.py               MIAConfig and the target/shadow/attack-model lifecycle
  partial_mia.py       Noise, masking, feature selection, enumeration helpers
  metrics.py           Confusion-matrix and rank metrics
  experiments.py       Dataset loaders, model factories, presets, JSON results
  run_experiment.py    Command-line runner for every dataset/model combination
  *_eval_partial_mia.py  Thin entry points matching the original script names
  tests/               Unit tests
legacy/                Original research scripts, kept unchanged for reference
  cnn/                 Standalone TensorFlow CNN drafts (CIFAR-10/100)
datasets/purchase/     Purchase-100 cluster labels (feature file downloaded separately)
docs/                  Report and code overview
requirements.txt       Core dependencies
requirements-image.txt Adds TensorFlow for the CIFAR-10 experiments
```

[docs/CODE_OVERVIEW.md](docs/CODE_OVERVIEW.md) describes the architecture, every defect fixed relative to `legacy/`, and the remaining research limitations.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the CIFAR-10 experiments, also install TensorFlow:

```bash
python -m pip install -r requirements-image.txt
```

Run all commands from the repository root.

## Data

| Dataset | Source |
| --- | --- |
| Purchase-100 | Cluster labels are in `datasets/purchase/`. The feature matrix must be downloaded separately (see below). |
| CIFAR-10 | Downloaded automatically by TensorFlow on first use. |
| 20 Newsgroups | Downloaded automatically by scikit-learn on first use. |

### Purchase-100 feature file

`modified_purchase100.npy` (947 MB) exceeds GitHub's file-size limit and is not stored in this repository.

**Download:** _link to be added_

Place it at `datasets/purchase/modified_purchase100.npy`. It is a pickled zero-dimensional NumPy object array whose `features` key holds an `int64` matrix of shape `(197324, 600)`. Because it is loaded with `allow_pickle=True`, only use a copy from a trusted source and check its hash:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `modified_purchase100.npy` | 947155588 | `fdbfd6b9f0b1551dda007655084718738c9f1ccbcbee79587788220fec10253b` |
| `cluster_labels_2.npy` | 789424 | `a3b11582d3f53d2625324cce5af6ba1f9ab7b6c12fecdb6b6d0bf6dada75a60c` |
| `cluster_labels_10.npy` | 789424 | `f756e19965bebb4694259685fddc186821c3e5eba6b1647a84493febcbd682d7` |
| `cluster_labels_20.npy` | 789424 | `3c88e3bf7c022b6b3c5857095e0b4b280d2343ee03868103bfa3dc725aa25d63` |
| `cluster_labels_50.npy` | 789424 | `483e64dab4a8f3923d02aee687e22a4c9c0f6a990aede897b0e21d85793f7756` |
| `cluster_labels_100.npy` | 789424 | `247376e737d71eaf0e7ede5b39f77e373cbb26170d36137d91217559755e3a4f` |

```bash
# Windows PowerShell
Get-FileHash -Algorithm SHA256 datasets\purchase\modified_purchase100.npy
# macOS/Linux
sha256sum datasets/purchase/modified_purchase100.npy
```

## Running experiments

Reduced Purchase / logistic-regression smoke run:

```bash
python -m pmia --modality purchase --model lr --smoke --output results/purchase-lr-smoke.json
```

`--smoke` results are labeled `SMOKE/REDUCED` in the output JSON and record the reduced dataset and sample sizes. They are not canonical results.

Full-data preset:

```bash
python -m pmia --modality purchase --model lr --output results/purchase-lr-full.json
```

The full Purchase run loads the 947 MB feature file, hashes it by default, and trains several models. It is not a quick test.

The six original experiment names are available as entry points that call the same runner:

```bash
python pmia/lr_binary_eval_partial_mia.py --smoke
python pmia/mlp_binary_eval_partial_mia.py --smoke
python pmia/lr_image_eval_partial_mia.py --smoke
python pmia/mlp_image_eval_partial_mia.py --smoke
python pmia/lr_text_eval_partial_mia.py --smoke
python pmia/mlp_text_eval_partial_mia.py --smoke
```

| Modality | Model | Training records/model | Testing records/model |
| --- | --- | ---: | ---: |
| Purchase-100 | Logistic regression | 2500 | 499 |
| Purchase-100 | MLP | 2500 | 2500 |
| CIFAR-10, flattened | Logistic regression | 1000 | 499 |
| CIFAR-10, flattened | MLP | 1000 | 1000 |
| 20 Newsgroups TF-IDF | Logistic regression | 1000 | 499 |
| 20 Newsgroups TF-IDF | MLP | 1000 | 1000 |

### Methods

The default run evaluates:

- `baseline`: exact, unmodified target records;
- `noise`: binary flips for Purchase, Gaussian noise for image/text features;
- `mask`: one deterministic random column mask shared by members and non-members;
- `feature_selection`: masks the least-informative columns, scored on shadow-training data only, with the same mask applied to both evaluation groups.

Select methods explicitly with `--methods`:

```bash
python -m pmia --modality purchase --model lr --smoke --methods baseline,noise
```

Binary enumeration is opt-in because its size grows exponentially:

```bash
python -m pmia --modality purchase --model lr --smoke --methods baseline,enumerate --evaluation-size 100 --enumeration-fraction 0.01 --max-output-mib 512
```

Enumeration averages the membership probability over every completion and returns one decision per original record. The allocation size is estimated first; the run raises `MemoryError` instead of silently shrinking the experiment.

See every option with:

```bash
python -m pmia --help
```

### Output

Runs print JSON and, with `--output`, save it atomically. The output includes the exact settings and seed, package versions, data paths, shapes, class counts and hashes, actual sample counts, target/shadow/per-class attack accuracy, confusion-matrix counts, and accuracy, precision, recall, specificity, FPR, attack advantage, ROC AUC, average precision, and TPR at 1% FPR. `--no-hash-data` skips data hashing and records that choice.

### Failure behavior

Runs stop with a nonzero exit code, rather than falling back silently, when:

- scikit-learn or TensorFlow is missing;
- a dataset or label file is missing;
- a model lacks `predict_proba` or fitted `classes_`, or its probability columns do not match the declared classes;
- a per-class attack has no member or non-member examples;
- a metric denominator is zero;
- a model raises a convergence warning;
- a transformation exceeds its memory limit;
- a label count, sample size, or feature index is invalid.

## Tests

The unit tests use a small deterministic classifier and do not need scikit-learn:

```bash
python -m unittest discover -s pmia/tests -v
```

They validate the MIA engine, not the scientific outcome of a full experiment.

## Legacy code

`legacy/` contains the original experiment scripts unchanged. They run from the repository root, for example `python legacy/lr_binary_eval_partial_mia.py`, and need NumPy, pandas, scikit-learn, and typing-extensions. They contain known defects, including identical shadow datasets and partial-noise metrics computed on unmodified data, so their output should not be treated as valid results. See [legacy/README.md](legacy/README.md) and [docs/CODE_OVERVIEW.md](docs/CODE_OVERVIEW.md).
