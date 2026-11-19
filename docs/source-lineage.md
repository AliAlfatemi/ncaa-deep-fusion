# Source lineage

Selected neural definitions were extracted from the original project without
executing its top-level dataset reads, plotting, or training cells.

The LSTM now declares batch_first=True to match its forward input layout. This prevents one game from being interpreted as a time step of another game. The correction changes the historical script behavior; old measurements are not asserted for this release.

The portable fusion interface adds input validation, arbitrary model-column
names, explicit row identifiers, finite nonnegative weights, stable handling
of ties, and explicit behavior for constant columns. These are release-level
engineering changes, not evidence of matching archival benchmark outputs.
The reference module retains the source's cognitive-diversity expressions.
Full protocol reproduction requires separately establishing authoritative data
preparation, model configuration, calibration, and evaluation splits.
