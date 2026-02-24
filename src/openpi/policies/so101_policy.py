import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO101/yellow-cube policy."""
    return {
        "observation/state": np.random.rand(6),  # 5 joints + gripper
        "observation/top_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the yellow cube and place it",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Converts SO101/yellow-cube dataset observations into canonical OpenPI inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        top_image = _parse_image(data["observation/top_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Pi0 / Pi0.5 expect three image slots. Duplicate the top camera for the third slot.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": top_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Converts canonical OpenPI outputs back to SO101/yellow-cube action format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}

