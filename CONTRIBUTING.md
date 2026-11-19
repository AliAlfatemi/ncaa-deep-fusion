# Contributing

Run `python -m pytest -q`, `python -m ncaa_fusion smoke`, and `python -m build`.
Preserve alignment across prediction columns and keep weight selection separate
from evaluation labels. Changes to model layout, tie handling, or normalization
need regression coverage. Do not commit input records, generated predictions,
results, weights, manuscript files, credentials, or machine-specific paths.
Preserve the complete paper citation and identify software contributions separately.
