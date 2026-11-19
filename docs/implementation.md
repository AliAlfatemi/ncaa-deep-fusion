# Implementation notes

## Neural models

The library provides DFFNN, Model1 (dense network), and Model4 (LSTM).
The LSTM uses `batch_first=True` to match its batch/time/feature input layout.
This corrects the earlier script's interpretation of independent games as time
steps, so results from the earlier layout require separate evaluation.

## Fusion interface

The CLI accepts aligned prediction columns with explicit row identifiers and
finite, nonnegative weights. Score combination supports optional per-column
min-max normalization; constant columns receive 0.5. Rank combination averages
tied ranks. Weights are supplied by the caller and should be selected using
independent validation data.

The reference module preserves the research cognitive-diversity expressions
with fixed model-column names. The CLI provides prediction combination;
dataset preparation, model training, and calibration belong to the calling
experiment pipeline.
