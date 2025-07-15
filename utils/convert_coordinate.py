# 从左下角为原点的1080p坐标系
# 转换为
# 以中心点为原点的Unity Camera Space
# 并完成scale的缩放 pixel to factor

import numpy as np

depth = 0.5
# Scaling factor for conversion from pixel to factor in unity camera space
scaling_factor = 0.1/338  

def convert_coordinates(position, depth=depth, scaling_factor=scaling_factor):
    """
    Convert position from 1080p camera space to Unity camera space.
    """
    view_position = position.copy()
    # Adjust to center of camera space
    view_position -= np.array([940, 637])
    # Adjust amplitude from image pixel space to Unity camera space
    view_position = view_position.astype(np.float32)
    view_position *= scaling_factor
    # Add depth to the z-axis
    view_position = np.append(view_position, depth)
    return view_position


