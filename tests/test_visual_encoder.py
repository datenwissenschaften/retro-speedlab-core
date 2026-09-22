import numpy as np
import pytest

from datenwissenschaften.vision.encoder import FixedVisualEncoder


def test_encode_accepts_grayscale_2d_observations():
    encoder = FixedVisualEncoder()
    observation = np.zeros((8, 8), dtype=np.uint8)

    features = encoder.encode(observation)

    assert len(features) == FixedVisualEncoder.output_size
    assert all(-1.0 <= value <= 1.0 for value in features)


def test_encode_accepts_single_channel_observations():
    encoder = FixedVisualEncoder()
    observation = np.zeros((1, 8, 8), dtype=np.uint8)

    features = encoder.encode(observation)

    assert len(features) == FixedVisualEncoder.output_size


def test_encode_converts_three_channel_rgb_observations_to_grayscale():
    encoder = FixedVisualEncoder()
    observation = np.zeros((3, 8, 8), dtype=np.uint8)
    observation[0] = 255

    features = encoder.encode(observation)

    assert len(features) == FixedVisualEncoder.output_size


def test_encode_rejects_an_unsupported_channel_count():
    encoder = FixedVisualEncoder()
    observation = np.zeros((2, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel-first RGB or grayscale"):
        encoder.encode(observation)


def test_encode_rejects_observations_with_the_wrong_number_of_dimensions():
    encoder = FixedVisualEncoder()
    observation = np.zeros((2, 3, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="Expected observation shape"):
        encoder.encode(observation)
