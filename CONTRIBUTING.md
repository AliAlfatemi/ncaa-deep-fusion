# Contributing

Run `python -m pytest -q`, `python -m ncaa_fusion smoke`, and `python -m build`.
Describe the changed behavior and validation in a focused pull request.

Preserve alignment across prediction columns and select weights independently
of evaluation labels. Changes to model layout, tie handling, or normalization
should include regression coverage using synthetic inputs. Retain attribution
and the complete paper citation.
