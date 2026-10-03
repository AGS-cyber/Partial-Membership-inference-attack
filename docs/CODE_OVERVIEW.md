# Code overview and fix map

## Isolation boundary

Only files inside `pmia/` belong to the corrected implementation. No `legacy/` source is imported at runtime. Dataset loading is read-only and resolves paths through the repository root.

The rewrite covers the shared MIA engine in `legacy/` and all six original experiment combinations. The original code is kept unchanged in `legacy/` for comparison.

The standalone TensorFlow drafts `legacy/cnn/CIFAR-10.py` and `legacy/cnn/CIFAR-100.py` have known defects (see [legacy/README.md](../legacy/README.md)); they are not represented as repaired by the flattened CIFAR-10 LR/MLP pipeline.

## Module map

| Path | Responsibility |
| --- | --- |
| `pmia/constants.py` | Explicit `OUT = 0` and `IN = 1` labels. |
| `pmia/datastructures.py` | Shape/length validation, stratified splitting and sampling, record identity, model validation/cloning, classifier collections. |
| `pmia/mia.py` | Label encoding, target/shadow allocation, model lifecycle, balanced class-specific attack construction, membership queries. |
| `pmia/partial_mia.py` | Deterministic feature selection, noise, masking, permutation, distance, and guarded enumeration. |
| `pmia/metrics.py` | Strict confusion-matrix and rank metrics without zero-denominator defaults. |
| `pmia/experiments.py` | Dataset loaders, model factories, presets, perturbation plans, provenance, and JSON results. |
| `pmia/run_experiment.py` | One CLI for every modality/model combination. |
| `pmia/*_eval_partial_mia.py` | Thin compatibility entry points; no duplicated experiment logic. |
| `pmia/tests/test_core.py` | Deterministic unit tests for corrected invariants and historical failures. |

## End-to-end flow

1. The loader validates the selected source and constructs `LabeledDataset`.
2. An optional `--dataset-limit` or `--smoke` reduction uses a recorded deterministic stratified sample.
3. `MIAEnvironment` encodes labels into a new dataset. The caller's labels are not mutated.
4. A randomized stratified split creates disjoint target and shadow partitions.
5. Each partition is split into stratified model-training and model-testing pools.
6. One target sample and multiple independently drawn shadow samples are created with a single advancing random generator.
7. The target and shadow estimators are cloned and trained.
8. Shadow posterior vectors are routed by the model's **predicted class**, matching the information available during `is_member()`.
9. For each predicted class, member and non-member posterior records are explicitly checked and balanced.
10. One binary attack estimator is trained per routed class.
11. Target records provide balanced, per-class attack evaluation data.
12. Membership queries process only classes present in the query batch and write results directly into their original row positions.
13. A single perturbation plan is applied consistently to the member and non-member evaluation groups.
14. Metrics are computed once through `metrics.py` and serialized with exact counts and provenance.

## Historical bug fixes

| Original behavior | Corrected behavior |
| --- | --- |
| Every shadow sample reused the same seed and was identical. | One advancing generator produces reproducible but distinct samples; exact duplicate record sets raise. |
| Target/shadow partitions were the leading and trailing source rows. | Target/shadow allocation is randomized and stratified. |
| `MIAEnvironment` mutated the caller's labels. | Labels are encoded into a new dataset and the original is tested for immutability. |
| Runtime invariants used `assert`. | Public/runtime contracts raise explicit typed exceptions. |
| `np.float_` prevented import under NumPy 2. | Public probability outputs use `np.float64`. |
| Probability column index was assumed to equal its label. | Every probability matrix is reordered through fitted `classes_`, and the exact class set is validated. |
| Attack training grouped by true class but inference routed by predicted class. | Both training and inference route by predicted class. |
| Empty class queries were sent through every attack model. | Only classes present in the query are evaluated. |
| Single-record/class-sparse queries could fail. | One-dimensional and class-sparse batches are explicitly supported and tested. |
| Query results were sorted by class and never unsorted. | Preallocated outputs are filled at original row indices. |
| Boolean results were built from float arrays. | Decisions are allocated as `np.bool_`; probabilities are `np.float64`. |
| `is_member()` could run before attack training. | Lifecycle violations raise `RuntimeError`. |
| Per-class attack data could be empty or one-sided. | Both `IN` and `OUT` are required and balanced for every routed class. |
| Partial-noise training metrics accidentally used clean training records. | Metrics consume the exact modified member and non-member matrices. |
| Text LR used different train/test noise levels in one comparison. | The same recorded noise rule and level is applied to both groups. |
| Training and testing labels independently selected different features. | Feature scores use shadow-training data only; one selected mask is shared by both evaluation groups. |
| Random perturbations mixed unseeded APIs. | Every method derives and records its own deterministic seed, so results do not depend on method ordering. |
| `closest_match_distance()` silently returned `None`. | No match raises `LookupError`. |
| Dense 20 Newsgroups TF-IDF could require tens of gigabytes. | TF-IDF remains CSR sparse throughout loading and compatible transformations. |
| Enumeration allocated exponential arrays without a guard. | Size is estimated in bytes and compared with an explicit limit before allocation. |
| Enumeration metrics counted every completion as a separate person. | Completion probabilities are aggregated back to one result per original record. |
| Precision/recall could divide by zero. | Undefined denominators raise `UndefinedMetricError`. |
| Six scripts duplicated hundreds of lines of calculations. | Six thin wrappers call one validated runner. |
| Importing an experiment started training/downloads. | All execution is behind CLI/main guards. |
| `setup.py` looked like package metadata but was only constants. | Compatibility constants remain explicit; installation instructions use requirements files. |
| Results were unstructured console text. | Results are strict JSON with settings, counts, hashes, versions, metrics, and scientific-scope warnings. |
| Convergence warnings could be overlooked. | Scikit-learn convergence and runtime warnings are promoted to failures during model training. |

## Deliberate methodology changes

Correcting the implementation changes the experiment. New results are not numerically comparable with old console output because:

- target/shadow allocation is randomized rather than positional;
- shadow samples are independent rather than identical;
- per-class attack data is balanced;
- routing uses predicted class consistently;
- member and non-member transformations use one shared feature plan;
- text remains sparse;
- enumeration is aggregated per original record;
- metrics use correct modified inputs and strict definitions.

These are recorded design choices, not hidden compatibility fallbacks.

## Data and model presets

The default sample-size presets preserve the original intent:

| Modality | Model | Training records/model | Testing records/model |
| --- | --- | ---: | ---: |
| Purchase-100 | Logistic regression | 2500 | 499 |
| Purchase-100 | MLP | 2500 | 2500 |
| CIFAR-10, flattened | Logistic regression | 1000 | 499 |
| CIFAR-10, flattened | MLP | 1000 | 1000 |
| 20 Newsgroups TF-IDF | Logistic regression | 1000 | 499 |
| 20 Newsgroups TF-IDF | MLP | 1000 | 1000 |

All presets retain a 50/50 target/shadow partition, 70/30 inner pool split, and two shadow models unless overridden.

Logistic regression now receives explicit iteration limits. MLP target/shadow models use `(100,)` hidden units except text, which keeps `(100, 50, 25)`; attack MLPs use `(100,)`. Estimator convergence warnings abort the run.

## Partial-feature definitions

- **Noise:** the same selected columns are used for members and non-members. Purchase values are independently flipped with the configured probability. Dense image features receive independent Gaussian noise with the configured standard deviation. Sparse text Gaussian noise explicitly inserts values into selected columns and applies a memory guard.
- **Mask:** one selected column set is applied to both groups. The default sentinel is `-1` for dense binary Purchase data and `0` for image/text data. Sparse inputs reject nonzero sentinels because they would densify the matrix.
- **Feature selection:** Purchase and image use mutual information calculated only from shadow-training data. Sparse nonnegative text uses chi-squared scores. The least-scoring fraction is masked identically in both groups.
- **Enumeration:** Purchase only. All binary values for the selected columns are queried, membership probabilities are averaged, and the aggregate is thresholded at `0.5`.

These operational definitions still require dissertation-level justification as attacker models; fixing code does not prove that a perturbation is the correct scientific definition of partial knowledge.

## Validation levels

### Completed

- Non-importing syntax parsing.
- Import of every fixed core module in a NumPy/SciPy environment.
- Deterministic unit tests using a local probability classifier.
- CLI help and preset construction.
- Explicit missing-dependency failure.
- Original-file hash comparison.

### Not established by core unit tests

- Accuracy or convergence of scikit-learn LR/MLP models on the real datasets.
- TensorFlow/CIFAR download and execution.
- Full-data runtime and peak memory.
- Paper-level statistics across seeds and target models.
- Novelty or validity of the research threat model.

## Remaining research limitations

1. One CLI invocation still trains one target model with one seed. A paper needs repeated independent target models and uncertainty aggregation.
2. Target and shadow estimators use the same configured random seed but different data. A study may additionally vary initialization seeds.
3. The fixed pipeline evaluates black-box posterior attacks; it does not yet include loss, entropy, confidence-threshold, LiRA, or imputation-based baselines.
4. Feature selection uses attacker-accessible shadow training data, but whether that knowledge is realistic must be justified.
5. Sparse Gaussian perturbation can still be expensive because selected zero entries become explicit stored values.
6. Purchase features require trusted pickle loading because of the existing artifact format.
7. The committed `cluster_labels_20/50/100.npy` files differ from the versions in the original project upload. Record data hashes with every result (the README lists the current ones).
8. A successful run establishes software execution, not dissertation validity or paper novelty.

## Recommended next scientific step

After installing a clean environment, run only the Purchase LR smoke preset first. If it passes, inspect the JSON counts, class-specific attack scores, and baseline metrics before enabling perturbations or full data. Then repeat a declared seed matrix and aggregate confidence intervals in a separate analysis layer rather than manually copying console numbers.
