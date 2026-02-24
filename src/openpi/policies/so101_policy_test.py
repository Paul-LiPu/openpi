import numpy as np

from openpi.models import model as _model
from openpi.policies import so101_policy


def test_parse_image_converts_float_chw_to_uint8_hwc():
    image = np.ones((3, 4, 5), dtype=np.float32) * 0.5

    parsed = so101_policy._parse_image(image)

    assert parsed.shape == (4, 5, 3)
    assert parsed.dtype == np.uint8
    assert np.all(parsed == 127)


def test_so101_inputs_maps_keys_and_duplicates_top_camera():
    top = np.random.randint(0, 255, size=(8, 8, 3), dtype=np.uint8)
    wrist = np.random.randint(0, 255, size=(3, 8, 8), dtype=np.uint8).astype(np.float32) / 255.0
    state = np.arange(6, dtype=np.float32)
    actions = np.ones((10, 6), dtype=np.float32)

    outputs = so101_policy.SO101Inputs(model_type=_model.ModelType.PI05)(
        {
            "observation/top_image": top,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "actions": actions,
            "prompt": "pick and place",
        }
    )

    assert outputs["state"] is state
    assert outputs["actions"] is actions
    assert outputs["prompt"] == "pick and place"
    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], top)
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"], top)
    assert outputs["image"]["left_wrist_0_rgb"].shape == (8, 8, 3)
    assert outputs["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert all(bool(v) for v in outputs["image_mask"].values())


def test_so101_outputs_truncates_to_6d():
    actions = np.random.randn(10, 32).astype(np.float32)

    outputs = so101_policy.SO101Outputs()({"actions": actions})

    assert outputs["actions"].shape == (10, 6)
    np.testing.assert_array_equal(outputs["actions"], actions[:, :6])
