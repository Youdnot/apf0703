import numpy as np
from typing import Tuple


def _align_mask_to_depth(mask: np.ndarray, depth_shape: Tuple[int, int]) -> np.ndarray:
    """Aligns a standard mask to the depth image shape.

    The standard mask is provided as a boolean array with shape (1920, 1080),
    which follows the (width, height) convention. Depth images typically use
    (height, width). This function transposes the mask when necessary so that
    it matches the depth image shape.

    Args:
      mask: A 2D boolean numpy array representing the mask.
      depth_shape: The expected (height, width) shape of the depth image.

    Returns:
      A boolean numpy array whose shape exactly matches `depth_shape`.

    Raises:
      ValueError: If the mask cannot be aligned to the depth image shape.
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    height, width = depth_shape

    if mask.shape == (height, width):
        return mask.astype(bool, copy=False)

    if mask.shape == (width, height):
        return mask.T.astype(bool)

    if mask.T.shape == (height, width):
        return mask.T.astype(bool)

    raise ValueError(
        f"Mask shape {mask.shape} is not compatible with depth shape {depth_shape}."
    )


def compute_average_depth(depth_image: np.ndarray, mask: np.ndarray) -> float:
    """Computes the average depth within the masked region.

    The mask is a standard boolean array of shape (1920, 1080) (width, height)
    or already aligned to the depth image shape (height, width). This function
    aligns the mask to the depth image if needed and computes the mean of depth
    values where the mask is True.

    Args:
      depth_image: A 2D numpy array of shape (height, width) containing depth values.
      mask: A 2D boolean numpy array, either shape (1920, 1080) or aligned with the
        depth image shape.

    Returns:
      The average depth within the mask as a float.

    Raises:
      ValueError: If inputs are invalid, shapes are incompatible, or the mask has no
        True values after alignment.
    """
    if depth_image.ndim != 2:
        raise ValueError("Depth image must be a 2D array with shape (height, width).")

    aligned_mask = _align_mask_to_depth(mask, depth_image.shape)

    if not np.any(aligned_mask):
        raise ValueError("Mask contains no True values.")

    masked_values = depth_image[aligned_mask]
    if masked_values.size == 0:
        raise ValueError("No depth values selected by the mask.")

    return float(np.mean(masked_values))


if __name__ == "__main__":
    # Simple test case
    height, width = 1080, 1920
    depth = np.arange(height * width, dtype=np.float32).reshape(height, width)

    # Standard mask: (width, height) = (1920, 1080)
    std_mask = np.zeros((width, height), dtype=bool)
    std_mask[100:200, 10:20] = True  # x in [100,200), y in [10,20)

    avg = compute_average_depth(depth, std_mask)

    # Compute expected result for validation
    expected = float(np.mean(depth[std_mask.T]))

    print(f"Average depth: {avg}")
    print(f"Expected:      {expected}")
    assert np.isclose(avg, expected), "Average depth does not match expected value."
    print("Test passed.")


