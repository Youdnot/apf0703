# import time
# from multiprocessing import Process, Lock, Condition, Manager, Event, set_start_method, current_process
# from typing import Optional

# class BoundedStack:
#     """A simple process-safe bounded LIFO stack using a Manager-backed list.

#     This container favors most-recent data. When full, it discards the oldest
#     element (bottom of stack) to keep latency low. It supports blocking pop/peek
#     semantics with an optional timeout.

#     Attributes:
#         maxsize: Maximum number of elements to keep. If <= 0, behaves as unbounded.
#     """

#     def __init__(self, maxsize: int = 2) -> None:
#         self.maxsize = maxsize
#         self._manager = Manager()
#         self._stack = self._manager.list()  # type: ignore[var-annotated]
#         self._mutex = Lock()
#         self._not_empty = Condition(self._mutex)

#     def push(self, item) -> None:
#         """Pushes an item onto the top of the stack, dropping oldest if full.

#         Args:
#             item: Any picklable Python object (e.g., numpy arrays, small dicts).
#         """
#         with self._mutex:
#             if self.maxsize > 0 and len(self._stack) >= self.maxsize:
#                 # Remove the oldest element to prioritize recency and reduce latency.
#                 self._stack.pop(0)
#             self._stack.append(item)
#             self._not_empty.notify()

#     def pop(self, block: bool = True, timeout: Optional[float] = None):
#         """Pops the most recent item from the stack.

#         Args:
#             block: If True, block until an item is available or timeout occurs.
#             timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

#         Returns:
#             The most recent item.

#         Raises:
#             TimeoutError: If blocking with a timeout and no item becomes available.
#             RuntimeError: If non-blocking and the stack is empty.
#         """
#         with self._mutex:
#             if not block:
#                 if len(self._stack) == 0:
#                     raise RuntimeError("pop from empty stack (non-blocking)")
#             elif timeout is None:
#                 while len(self._stack) == 0:
#                     self._not_empty.wait()
#             else:
#                 end_time = time.time() + timeout
#                 while len(self._stack) == 0:
#                     remaining = end_time - time.time()
#                     if remaining <= 0:
#                         raise TimeoutError("pop timed out waiting for item")
#                     self._not_empty.wait(remaining)

#             return self._stack.pop()

#     def peek(self, block: bool = True, timeout: Optional[float] = None):
#         """Returns the most recent item without removing it.

#         Args:
#             block: If True, block until an item is available or timeout occurs.
#             timeout: Maximum time to wait in seconds if blocking. None means wait indefinitely.

#         Returns:
#             The most recent item.

#         Raises:
#             TimeoutError: If blocking with a timeout and no item becomes available.
#             RuntimeError: If non-blocking and the stack is empty.
#         """
#         with self._mutex:
#             if not block:
#                 if len(self._stack) == 0:
#                     raise RuntimeError("peek from empty stack (non-blocking)")
#             elif timeout is None:
#                 while len(self._stack) == 0:
#                     self._not_empty.wait()
#             else:
#                 end_time = time.time() + timeout
#                 while len(self._stack) == 0:
#                     remaining = end_time - time.time()
#                     if remaining <= 0:
#                         raise TimeoutError("peek timed out waiting for item")
#                     self._not_empty.wait(remaining)

#             return self._stack[-1]

#     def empty(self) -> bool:
#         with self._mutex:
#             return len(self._stack) == 0

#------------------------------------------------------------------------------
import numpy as np
import open3d as o3d
import numba as nb

import hl2ss
import hl2ss_3dcv

#------------------------------------------------------------------------------
# Optimization functions
def zero_order_hold(pv_list, pv_height, pv_width):
    """
    Unified zero-order hold implementation that combines the best of both approaches.
    
    Args:
        pv_list: numpy array with shape (N, 5) containing [u0, v0, u1, v1, depth]
        pv_height: height of PV image
        pv_width: width of PV image
    
    Returns:
        pv_z: depth map for PV image
    """
    pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)
    
    if pv_list.shape[0] == 0:
        return pv_z
    
    # Extract and clip coordinates in one step (combines both approaches)
    u0 = np.clip(pv_list[:, 0].astype(np.int32), 0, pv_width-1)
    v0 = np.clip(pv_list[:, 1].astype(np.int32), 0, pv_height-1)
    u1 = np.clip(pv_list[:, 2].astype(np.int32), 0, pv_width)
    v1 = np.clip(pv_list[:, 3].astype(np.int32), 0, pv_height)
    depth_values = pv_list[:, 4]
    
    # Vectorized validity check (only check for positive depth and valid rectangles)
    valid_mask = (depth_values > 0) & (u1 > u0) & (v1 > v0)
    
    if not np.any(valid_mask):
        return pv_z
    
    # Filter to valid entries only
    u0_valid = u0[valid_mask]
    v0_valid = v0[valid_mask]
    u1_valid = u1[valid_mask]
    v1_valid = v1[valid_mask]
    depth_valid = depth_values[valid_mask]
    
    # Optimized loop - only process valid rectangles
    for i in range(len(u0_valid)):
        pv_z[v0_valid[i]:v1_valid[i], u0_valid[i]:u1_valid[i]] = depth_valid[i]
    
    return pv_z

# optimized depth to pv rgb
@nb.jit(nopython=True, cache=True)
def numba_zero_order_hold(pv_list, pv_height, pv_width):
    """使用numba优化的零阶保持实现"""
    pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)
    
    for n in range(pv_list.shape[0]):
        u0 = max(0, min(int(pv_list[n, 0]), pv_width-1))
        v0 = max(0, min(int(pv_list[n, 1]), pv_height-1))
        u1 = max(0, min(int(pv_list[n, 2]), pv_width))
        v1 = max(0, min(int(pv_list[n, 3]), pv_height))
        depth = pv_list[n, 4]
        
        if depth > 0 and u1 > u0 and v1 > v0:
            for v in range(v0, v1):
                for u in range(u0, u1):
                    pv_z[v, u] = depth
    
    return pv_z

# optimize eet projection
def optimized_mesh_generation(depth):
    """
    Optimized mesh generation using numpy vectorization instead of nested for loops.
    
    Args:
        depth: depth image array
        world_points: 3D world coordinates for each depth pixel
    
    Returns:
        faces: list of triangle face indices
    """
    h, w = depth.shape[-2:]
    mask = depth > 0
    
    # Create coordinate grids using numpy
    i_coords, j_coords = np.meshgrid(np.arange(1, h), np.arange(1, w), indexing='ij')
    
    # Flatten coordinates for vectorized operations
    i_flat = i_coords.flatten()
    j_flat = j_coords.flatten()
    
    # Calculate all corner coordinates at once
    ul_i = (i_flat - 1) * w + (j_flat - 1)  # upper-left
    ur_i = (i_flat - 1) * w + j_flat        # upper-right  
    bl_i = i_flat * w + (j_flat - 1)        # bottom-left
    br_i = i_flat * w + j_flat              # bottom-right
    
    # Get validity masks for all corners
    ul_valid = mask[i_flat - 1, j_flat - 1]
    ur_valid = mask[i_flat - 1, j_flat]
    bl_valid = mask[i_flat, j_flat - 1]
    br_valid = mask[i_flat, j_flat]
    
    # Create triangles using vectorized conditions
    faces = []
    
    # Triangle 1: bottom-left, bottom-right, upper-right
    triangle1_mask = bl_valid & br_valid & ur_valid
    triangle1_indices = np.column_stack((bl_i[triangle1_mask], 
                                        br_i[triangle1_mask], 
                                        ur_i[triangle1_mask]))
    
    # Triangle 2: upper-right, upper-left, bottom-left  
    triangle2_mask = ur_valid & ul_valid & bl_valid
    triangle2_indices = np.column_stack((ur_i[triangle2_mask], 
                                        ul_i[triangle2_mask], 
                                        bl_i[triangle2_mask]))
    
    # Combine all triangles
    if len(triangle1_indices) > 0:
        faces.extend(triangle1_indices.tolist())
    if len(triangle2_indices) > 0:
        faces.extend(triangle2_indices.tolist())
    
    return faces

def create_raycast_scene_optimized(depth, world_points):
    """
    Create raycast scene with optimized mesh generation.
    """
    faces = optimized_mesh_generation(depth)

    vertices = o3d.core.Tensor(np.asarray(world_points.reshape(-1, 3), dtype=np.float32))
    triangles = o3d.core.Tensor(np.asarray(faces, dtype=np.int32))
    mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
    rcs = o3d.t.geometry.RaycastingScene()
    rcs.add_triangles(mesh)
    return rcs
    

# 使用 numba JIT 编译优化关键计算
@nb.jit(nopython=True, cache=True)
def fast_transform_and_project(xy1_o, xy1_d, z, lt_to_world, world_to_pv, pv_to_pv_image):
    """使用 numba 加速的变换和投影操作"""
    # 提取有效深度区域
    z_crop = z[:-1, :-1, :]
    
    # 计算点云
    lt_points_o = xy1_o * z_crop
    lt_points_d = xy1_d * z_crop
    
    # 变换到世界坐标系
    world_points_o = transform_points_nb(lt_points_o, lt_to_world)
    world_points_d = transform_points_nb(lt_points_d, lt_to_world)
    
    # 变换到PV坐标系并投影
    pv_points_o = transform_points_nb(world_points_o, world_to_pv)
    pv_depth = pv_points_o[:, :, 2:3]
    
    pv_uv_o = project_points_nb(pv_points_o, pv_to_pv_image)
    pv_uv_d = project_points_nb(world_points_d, world_to_pv @ pv_to_pv_image)
    
    return pv_uv_o, pv_uv_d, pv_depth

@nb.jit(nopython=True, cache=True)
def transform_points_nb(points, transform_matrix):
    """numba 优化的点变换"""
    h, w, _ = points.shape
    result = np.zeros_like(points)
    
    for i in range(h):
        for j in range(w):
            point = points[i, j]
            # 齐次坐标变换
            result[i, j, 0] = transform_matrix[0, 0] * point[0] + transform_matrix[0, 1] * point[1] + transform_matrix[0, 2] * point[2] + transform_matrix[0, 3]
            result[i, j, 1] = transform_matrix[1, 0] * point[0] + transform_matrix[1, 1] * point[1] + transform_matrix[1, 2] * point[2] + transform_matrix[1, 3]
            result[i, j, 2] = transform_matrix[2, 0] * point[0] + transform_matrix[2, 1] * point[1] + transform_matrix[2, 2] * point[2] + transform_matrix[2, 3]
    
    return result

@nb.jit(nopython=True, cache=True)
def project_points_nb(points_3d, projection_matrix):
    """numba 优化的点投影"""
    h, w, _ = points_3d.shape
    result = np.zeros((h, w, 2), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            point = points_3d[i, j]
            if point[2] > 0:  # 避免除零
                result[i, j, 0] = projection_matrix[0, 0] * point[0] / point[2] + projection_matrix[0, 2]
                result[i, j, 1] = projection_matrix[1, 1] * point[1] / point[2] + projection_matrix[1, 2]
    
    return result