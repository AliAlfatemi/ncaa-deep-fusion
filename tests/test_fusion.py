import numpy as np
import pandas as pd
import pytest
from ncaa_fusion.fusion import combine_frame, fuse_scores


def test_fixed_weighted_scores_match_hand_computation():
    np.testing.assert_allclose(
        fuse_scores([[0.2, 0.8], [0.6, 0.4]], [1, 3]), [0.65, 0.45]
    )


def test_rank_combination_is_invariant_to_increasing_affine_scores():
    x = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
    np.testing.assert_allclose(
        fuse_scores(x, method="rank"),
        fuse_scores(x * [2.0, 5.0] + [7.0, 9.0], method="rank"),
    )


def test_ties_and_constant_columns_have_defined_behavior():
    np.testing.assert_allclose(fuse_scores([[1, 2], [1, 2]], method="rank"), [0.5, 0.5])
    np.testing.assert_allclose(
        fuse_scores([[1, 2], [1, 2]], normalize=True), [0.5, 0.5]
    )


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [1]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        fuse_scores([[0.1, 0.2]], weights)


def test_duplicate_or_missing_sample_ids_are_rejected():
    for ids in [["a", "a"], ["a", None]]:
        with pytest.raises(ValueError, match="unique"):
            combine_frame(pd.DataFrame({"ID": ids, "m": [0.1, 0.2]}), ["m"])


def test_evaluation_labels_do_not_influence_fusion():
    f = pd.DataFrame({"ID": ["a", "b"], "m": [0.1, 0.9], "truth": [0, 1]})
    first = combine_frame(f, ["m"])
    f["truth"] = [1, 0]
    pd.testing.assert_frame_equal(first, combine_frame(f, ["m"]))
