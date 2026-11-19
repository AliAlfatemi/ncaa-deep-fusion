# NCAA Deep Learning and Fusion

[![CI](https://github.com/AliAlfatemi/ncaa-deep-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAlfatemi/ncaa-deep-fusion/actions/workflows/ci.yml)

Neural model definitions and score/rank combination components extracted from
the research project, with a portable fusion command and synthetic tests.

**Paper:** [Advancing NCAA March Madness Forecasts Through Deep Learning and Combinatorial Fusion Analysis](https://doi.org/10.1007/978-3-031-66431-1_38)

**Authors:** Ali Alfatemi, Mohamed Rahouti, D. Frank Hsu, Christina Schweikert

## Install and validate

```bash
git clone https://github.com/AliAlfatemi/ncaa-deep-fusion.git
cd ncaa-deep-fusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ncaa_fusion smoke
```

## Combine aligned predictions

Supply one CSV containing a unique `ID` and numeric score columns. Each row
must represent the same item across all model columns, and higher scores must
consistently mean greater positive-class confidence.

```bash
python -m ncaa_fusion fuse --csv /path/to/aligned-predictions.csv \
  --columns model_a model_b model_c model_d \
  --weights 1 1 1 1 --method score --output outputs/fusion-001
```

Use `--method rank` for average-rank combination with tied ranks. Use
`--normalize` to min-max scale each score column within the supplied batch;
constant columns receive a neutral value of 0.5. Weights must be finite,
nonnegative, and have positive total. Choose weights using independent
validation data; the command does not learn weights from evaluation labels.
It writes fused scores/ranks and method metadata to a fresh local directory.

## Library components

- `ncaa_fusion.models`: DFFNN, Model1 (dense network), and Model4 (LSTM).
- `ncaa_fusion.fusion`: validated score/rank combination for aligned arrays/data frames.
- `ncaa_fusion.reference`: original fixed-column cognitive-diversity calculations.

The model classes are available as a PyTorch library. The command combines
predictions; it does not implement full dataset preparation, backbone training,
model selection, or the complete published experiment grid. Reference diversity
functions retain their original fixed model-column names and are not automatically
used as calibrated weights by the command.

## Source lineage and release scope

See [source lineage](docs/source-lineage.md). This is a maintained component
release, not a byte-identical archive of a publication run. No published metric
has been re-established by its synthetic tests. Source data, exported predictions,
results, checkpoints, notebooks with outputs, and manuscript assets are excluded.

## Development, citation, and license

```bash
python -m pytest -q
python -m build
```

CI checks Python 3.12/3.13, model forward passes, fusion behavior, and the installed
wheel. [CITATION.cff](CITATION.cff) contains the complete published author order.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE).
