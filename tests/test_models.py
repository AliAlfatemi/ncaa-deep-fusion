import numpy as np
import pandas as pd
import torch
from ncaa_fusion.models import DFFNN, Model1, Model4
from ncaa_fusion.reference import cognitiveD, diversityS

torch.set_num_threads(1)


def test_extracted_models_accept_batches_of_prepared_game_features():
    for model in [DFFNN(8, 16, 1), Model1(8, 0.4), Model4(8)]:
        model.eval()
        out = model(torch.randn(5, 8))
        assert out.shape == (5, 1) and bool(torch.isfinite(out).all())
        out.mean().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_lstm_predictions_do_not_depend_on_game_order():
    model = Model4(8).eval()
    X = torch.randn(5, 8)
    order = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        original = model(X)
        permuted = model(X[order])
    torch.testing.assert_close(permuted, original[order])


def test_reference_diversity_preserves_five_model_symmetry():
    data = pd.DataFrame({f"Model{i}": [0.1 * i, 0.15 * i] for i in range(1, 6)})
    d = cognitiveD(data).to_numpy(dtype=float)
    np.testing.assert_allclose(d, d.T)
    np.testing.assert_allclose(np.diag(d), 0)
    assert len(diversityS(data)) == 5
