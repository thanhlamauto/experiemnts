from __future__ import annotations

import math
import ssl
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError

import numpy as np
import torch

from analyze_dinov2_spatial_norm import (
    _looks_like_ssl_verification_error,
    compute_similarity_maps,
    find_local_flax_pickle,
    grid_size_from_tokens,
    infer_dinov2_config_kwargs_from_name,
    model_spec_to_local_path,
    normalized_anchor_to_grid,
    parse_anchor_points,
    resolve_anchor_specs,
    spatial_normalize_tokens,
)


class Dinov2SpatialNormExperimentTests(unittest.TestCase):
    def test_spatial_normalize_matches_formula(self) -> None:
        tokens = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]],
                [[2.0, 1.0], [4.0, 5.0], [6.0, 9.0]],
            ]
        )

        expected = tokens - 0.7 * tokens.mean(dim=1, keepdim=True)
        expected = expected / (expected.std(dim=1, keepdim=True) + 1e-6)

        actual = spatial_normalize_tokens(tokens, gamma=0.7)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_parse_anchor_points_defaults(self) -> None:
        anchors = parse_anchor_points(None)
        self.assertEqual(len(anchors), 4)
        self.assertTrue(all(len(anchor) == 2 for anchor in anchors))

    def test_resolve_anchor_specs_reuses_single_spec(self) -> None:
        anchor_sets = resolve_anchor_specs(["0.1,0.2;0.9,0.8"], num_images=3)
        self.assertEqual(len(anchor_sets), 3)
        self.assertEqual(anchor_sets[0], anchor_sets[1])
        self.assertEqual(anchor_sets[1], anchor_sets[2])

    def test_normalized_anchor_to_grid_clips_and_rounds(self) -> None:
        self.assertEqual(normalized_anchor_to_grid((0.0, 0.0), 16, 16), (0, 0))
        self.assertEqual(normalized_anchor_to_grid((1.0, 1.0), 16, 16), (15, 15))
        self.assertEqual(normalized_anchor_to_grid((-1.0, 2.0), 16, 16), (15, 0))
        self.assertEqual(normalized_anchor_to_grid((0.49, 0.51), 4, 4), (2, 1))

    def test_grid_size_from_tokens_requires_square_grid(self) -> None:
        self.assertEqual(grid_size_from_tokens(256), (16, 16))
        with self.assertRaises(ValueError):
            grid_size_from_tokens(250)

    def test_compute_similarity_maps_has_unit_anchor_response(self) -> None:
        tokens = torch.eye(4, dtype=torch.float32)
        maps = compute_similarity_maps(tokens, anchors_rc=[(0, 0)], grid_width=2, grid_height=2)

        self.assertEqual(len(maps), 1)
        self.assertEqual(maps[0].shape, (2, 2))
        self.assertTrue(math.isclose(float(maps[0][0, 0]), 1.0, rel_tol=0.0, abs_tol=1e-6))
        self.assertTrue(np.allclose(maps[0][0, 1:], 0.0))

    def test_ssl_verification_error_detection(self) -> None:
        direct = ssl.SSLCertVerificationError("hostname mismatch")
        wrapped = URLError(direct)
        self.assertTrue(_looks_like_ssl_verification_error(direct))
        self.assertTrue(_looks_like_ssl_verification_error(wrapped))

    def test_infer_dinov2_config_kwargs_from_name_for_vitb14(self) -> None:
        kwargs = infer_dinov2_config_kwargs_from_name("dinov2_vitb14_flax.pkl")
        self.assertEqual(kwargs["hidden_size"], 768)
        self.assertEqual(kwargs["num_hidden_layers"], 12)
        self.assertEqual(kwargs["num_attention_heads"], 12)
        self.assertEqual(kwargs["patch_size"], 14)

    def test_find_local_flax_pickle_in_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "dinov2_vitb14_flax.pkl"
            target.write_bytes(b"pickle")
            (root / "notes.txt").write_text("x")
            found = find_local_flax_pickle(root)
            self.assertEqual(found, target)

    def test_model_spec_to_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.assertEqual(model_spec_to_local_path(str(root)), root)
        self.assertIsNone(model_spec_to_local_path("facebook/dinov2-base"))


if __name__ == "__main__":
    unittest.main()
