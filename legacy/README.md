# Legacy code

The original research scripts, kept unchanged for reference and comparison. The corrected implementation is in [`pmia/`](../pmia); [docs/CODE_OVERVIEW.md](../docs/CODE_OVERVIEW.md) maps each defect below to its fix.

| Path | Role |
| --- | --- |
| `datastructures.py` | Dataset wrappers, train/test containers, classifier protocol and collection |
| `mia.py` | `MIAConfig` and the target/shadow/attack-model lifecycle |
| `partial_mia.py` | Feature enumeration, distance, permutation, flip and Gaussian noise helpers |
| `setup.py` | Constants (`IN`, `OUT` and unused hyperparameters), not packaging metadata |
| `lr_binary_eval_partial_mia.py` | Purchase-100 (2 clusters), logistic regression |
| `mlp_binary_eval_partial_mia.py` | Purchase-100 (2 clusters), MLP |
| `lr_image_eval_partial_mia.py` | Flattened CIFAR-10, logistic regression |
| `mlp_image_eval_partial_mia.py` | Flattened CIFAR-10, MLP |
| `lr_text_eval_partial_mia.py` | 20 Newsgroups TF-IDF (dense), logistic regression |
| `mlp_text_eval_partial_mia.py` | 20 Newsgroups TF-IDF (dense), MLP |
| `cnn/CIFAR-10.py`, `cnn/CIFAR-100.py` | Standalone TensorFlow CNN shadow-model drafts |

## Running

Run from the repository root so the relative `datasets/...` paths resolve:

```bash
python legacy/lr_binary_eval_partial_mia.py
```

Each script runs a full experiment at import time (no `main()` guard). The image and text scripts download CIFAR-10 or 20 Newsgroups, and the text scripts densify the full TF-IDF matrix, which can need many gigabytes of memory.

## Known defects

Results produced by this code should not be treated as valid.

1. `TrainTestSplit.sample_n()` reuses the same `random_state` on every iteration, so all shadow datasets are identical.
2. `MIAEnvironment.is_member()` sorts inputs by predicted class and never restores the original row order.
3. `is_member()` calls every class-specific attack model, including classes absent from the query, so small or single-record batches can fail.
4. `is_member()` does not check that attack models have been trained.
5. Membership decisions are stored in a float array rather than a boolean one.
6. `np.float_` in a type annotation prevents importing `mia.py` under NumPy 2.
7. Attack models are trained by true class but queried by predicted class, so misclassified records reach the wrong attack model.
8. `predict_proba(...)[:, IN]` assumes the membership class is column 1 without checking `classes_`.
9. Per-class attack datasets are not checked for emptiness or for containing both `IN` and `OUT`.
10. Runtime validation uses `assert`, which `python -O` removes.
11. Target/shadow allocation uses source row order before stratification.
12. `MIAEnvironment` label-encodes the caller's dataset in place.
13. Feature selection and perturbation mix unseeded random generators, so results are not reproducible.
14. `closest_match_distance()` returns `None` when no match exists.
15. In all six evaluation scripts, the partial-noise training metrics are recomputed from the unmodified training data.
16. `lr_text_eval_partial_mia.py` uses noise level 0.1 for training records but 0.5 for testing records in the same comparison.
17. Mutual-information features are selected separately with training and testing labels, leaking test labels and applying different masks to each group.
18. Several precision/recall calculations can divide by zero.
19. Metrics over enumerated feature combinations count generated variants, not original records.

`cnn/` drafts:

- `CIFAR-10.py` was adapted from the 100-class version: it builds 20 attack features but uses 200-dimensional attack inputs and slices columns `100:200`.
- Shadow models train on `train_test_split(..., random_state=i)` partitions, but attack membership labels come from a separate fixed `random_state=42` split, so they do not describe most shadow models' actual membership.
- `CIFAR-100.py` reports `model.evaluate(...)[0]` (the loss) as accuracy.
- The `learning_rate` argument of `create_target_model` is never applied.
- Each script trains one target CNN, 20 shadow CNNs and many attack networks for hundreds of epochs.
