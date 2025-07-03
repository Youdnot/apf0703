# cursor 重组织的代码，部份类有重构，可以运行，但行为不如预期
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional, Dict, Any


class ObstacleManager:
    """Manages obstacle representation and operations for APF navigation.
    
    This class handles the conversion between segmentation masks and obstacle
    points, provides efficient overlap detection, and manages dynamic obstacle
    updates for AR window applications.
    """
    
    def __init__(self, image_shape: Tuple[int, int] = (1080, 1920)):
        """Initialize the obstacle manager.
        
        Args:
            image_shape: Tuple of (height, width) for the image dimensions.
        """
        self.image_shape = image_shape
        self.obstacle_masks: List[np.ndarray] = []
        self.obstacle_bboxes: List[List[float]] = []
        
    def add_obstacle_from_mask(self, mask: np.ndarray, bbox: Optional[List[float]] = None) -> None:
        """Add an obstacle from a segmentation mask.
        
        Args:
            mask: Boolean array of shape (height, width) where True indicates obstacle.
            bbox: Optional bounding box [x1, y1, x2, y2] for the mask.
        """
        # Resize mask to match image shape if necessary
        if mask.shape != self.image_shape:
            # Simple resize by cropping or padding to match target shape
            target_height, target_width = self.image_shape
            mask_height, mask_width = mask.shape
            
            # Create new mask with target shape
            resized_mask = np.zeros(self.image_shape, dtype=bool)
            
            # Copy overlapping region
            h_start = max(0, (target_height - mask_height) // 2)
            w_start = max(0, (target_width - mask_width) // 2)
            h_end = min(target_height, h_start + mask_height)
            w_end = min(target_width, w_start + mask_width)
            
            mask_h_start = max(0, (mask_height - target_height) // 2)
            mask_w_start = max(0, (mask_width - target_width) // 2)
            mask_h_end = min(mask_height, mask_h_start + (h_end - h_start))
            mask_w_end = min(mask_width, mask_w_start + (w_end - w_start))
            
            resized_mask[h_start:h_end, w_start:w_end] = mask[mask_h_start:mask_h_end, mask_w_start:mask_w_end]
            mask = resized_mask
        
        self.obstacle_masks.append(mask.copy())
        
        if bbox is None:
            # Calculate bounding box from mask
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                bbox = [float(x_coords.min()), float(y_coords.min()), 
                       float(x_coords.max()), float(y_coords.max())]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
        
        self.obstacle_bboxes.append(bbox)
    
    def create_test_obstacle_mask(self, center_x: int, center_y: int, 
                                 width: int, height: int) -> np.ndarray:
        """Create a simple rectangular obstacle mask for testing.
        
        Args:
            center_x: Center x coordinate of the obstacle.
            center_y: Center y coordinate of the obstacle.
            width: Width of the obstacle.
            height: Height of the obstacle.
            
        Returns:
            Boolean mask array where True indicates obstacle.
        """
        mask = np.zeros(self.image_shape, dtype=bool)
        
        # Calculate bounds
        x1 = max(0, center_x - width // 2)
        x2 = min(self.image_shape[1], center_x + width // 2)
        y1 = max(0, center_y - height // 2)
        y2 = min(self.image_shape[0], center_y + height // 2)
        
        # Set obstacle region
        mask[y1:y2, x1:x2] = True
        
        return mask
    
    def get_obstacle_points_in_window(self, window_bbox: List[float], 
                                    sample_density: int = 5) -> np.ndarray:
        """Get obstacle points that overlap with the given window.
        
        Args:
            window_bbox: Window bounding box [x1, y1, x2, y2].
            sample_density: Number of pixels to skip when sampling (for performance).
            
        Returns:
            Array of obstacle points (N, 2) in window coordinates.
        """
        window_x1, window_y1, window_x2, window_y2 = window_bbox
        obstacle_points = []
        
        for i, mask in enumerate(self.obstacle_masks):
            obs_bbox = self.obstacle_bboxes[i]
            
            # Check if obstacle bbox overlaps with window
            if (obs_bbox[0] < window_x2 and obs_bbox[2] > window_x1 and
                obs_bbox[1] < window_y2 and obs_bbox[3] > window_y1):
                
                # Calculate intersection region
                x_start = max(int(obs_bbox[0]), int(window_x1))
                x_end = min(int(obs_bbox[2]), int(window_x2))
                y_start = max(int(obs_bbox[1]), int(window_y1))
                y_end = min(int(obs_bbox[3]), int(window_y2))
                
                # Extract obstacle points in intersection region
                intersection_mask = mask[y_start:y_end:sample_density, 
                                       x_start:x_end:sample_density]
                y_coords, x_coords = np.where(intersection_mask)
                
                if len(y_coords) > 0:
                    # Convert to original image coordinates
                    x_coords = x_coords * sample_density + x_start
                    y_coords = y_coords * sample_density + y_start
                    points = np.column_stack([x_coords, y_coords])
                    obstacle_points.append(points)
        
        if obstacle_points:
            return np.vstack(obstacle_points)
        return np.empty((0, 2))
    
    def clear_obstacles(self) -> None:
        """Clear all stored obstacles."""
        self.obstacle_masks.clear()
        self.obstacle_bboxes.clear()


class APFCalculator:
    """Artificial Potential Field force calculator.
    
    This class provides unified force calculations for both attractive and
    repulsive forces, supporting both point obstacles and mask-based obstacles.
    """
    
    def __init__(self, k_att: float = 10.0, k_rep: float = 20.0, d0: float = 200.0,
                 max_att_force: float = 50.0, max_rep_force: float = 50.0):
        """Initialize the APF calculator.
        
        Args:
            k_att: Attractive force coefficient.
            k_rep: Repulsive force coefficient.
            d0: Obstacle influence range.
            max_att_force: Maximum attractive force magnitude.
            max_rep_force: Maximum repulsive force magnitude.
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d0 = d0
        self.max_att_force = max_att_force
        self.max_rep_force = max_rep_force
    
    def attractive_force(self, position: np.ndarray, goal: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate attractive force towards the goal.
        
        Args:
            position: Current position (2,).
            goal: Goal position (2,).
            
        Returns:
            Tuple of (unit_direction, magnitude).
        """
        direction = goal - position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            unit_direction = direction / distance
            magnitude = min(self.k_att * distance, self.max_att_force)
        else:
            unit_direction = np.zeros(2)
            magnitude = 0.0
            
        return unit_direction, magnitude
    
    def repulsive_force(self, position: np.ndarray, obstacles: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calculate repulsive force from obstacles using tensor operations.
        
        Args:
            position: Current position (2,).
            obstacles: Array of obstacle points (N, 2).
            
        Returns:
            Tuple of (unit_direction, magnitude).
        """
        if len(obstacles) == 0:
            return np.zeros(2), 0.0
        
        # Vectorized distance calculation
        diff = position.reshape(1, 2) - obstacles  # (N, 2)
        distances = np.linalg.norm(diff, axis=1)  # (N,)
        
        # Filter obstacles within influence range
        valid_mask = (distances < self.d0) & (distances > 0)
        
        if not np.any(valid_mask):
            return np.zeros(2), 0.0
        
        # Calculate repulsive force for valid obstacles
        valid_distances = distances[valid_mask]
        valid_diff = diff[valid_mask]
        
        # Repulsive force formula: k_rep * (1/d - 1/d0) * (1/d^2)
        force_coeffs = self.k_rep * ((1.0 / valid_distances) - (1.0 / self.d0)) * (1.0 / valid_distances ** 2)
        
        # Apply force coefficients to direction vectors
        repulsive_forces = force_coeffs[:, np.newaxis] * valid_diff  # (M, 2)
        
        # Sum all repulsive forces
        total_repulsive_force = np.sum(repulsive_forces, axis=0)
        
        # Calculate magnitude
        magnitude = np.linalg.norm(total_repulsive_force)
        
        if magnitude > 0:
            # Limit maximum repulsive force
            if magnitude > self.max_rep_force:
                magnitude = self.max_rep_force
                # Recalculate unit direction based on limited magnitude
                unit_direction = total_repulsive_force / np.linalg.norm(total_repulsive_force)
            else:
                unit_direction = total_repulsive_force / magnitude
        else:
            unit_direction = np.zeros(2)
            
        return unit_direction, magnitude
    
    def find_potential_minimum(self, anchor_point: np.ndarray, obstacles: np.ndarray, 
                              search_radius: float = 300.0, grid_size: int = 20) -> np.ndarray:
        """Find the minimum potential point in the field.
        
        Args:
            anchor_point: The anchor point (user's preferred position).
            obstacles: Array of obstacle points (N, 2).
            search_radius: Radius around anchor point to search for minimum.
            grid_size: Number of grid points for search.
            
        Returns:
            Position of minimum potential.
        """
        if len(obstacles) == 0:
            return anchor_point.copy()
        
        # Create search grid around anchor point
        x_range = np.linspace(anchor_point[0] - search_radius, anchor_point[0] + search_radius, grid_size)
        y_range = np.linspace(anchor_point[1] - search_radius, anchor_point[1] + search_radius, grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        
        min_potential = float('inf')
        min_position = anchor_point.copy()
        
        # Search for minimum potential
        for i in range(grid_size):
            for j in range(grid_size):
                position = np.array([X[i, j], Y[i, j]])
                
                # Calculate attractive potential
                dist_to_anchor = np.linalg.norm(position - anchor_point)
                att_potential = 0.5 * self.k_att * dist_to_anchor ** 2
                
                # Calculate repulsive potential
                rep_potential = 0.0
                for obs in obstacles:
                    dist_to_obs = np.linalg.norm(position - obs)
                    if dist_to_obs < self.d0 and dist_to_obs > 0:
                        rep_potential += 0.5 * self.k_rep * ((1.0 / dist_to_obs) - (1.0 / self.d0)) ** 2
                
                total_potential = att_potential + rep_potential
                
                if total_potential < min_potential:
                    min_potential = total_potential
                    min_position = position.copy()
        
        return min_position
    
    def total_force(self, position: np.ndarray, goal: np.ndarray, 
                   obstacles: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Calculate total force combining attractive and repulsive components.
        
        Args:
            position: Current position (2,).
            goal: Goal position (2,).
            obstacles: Array of obstacle points (N, 2).
            
        Returns:
            Tuple of (total_force, total_magnitude, force_info).
        """
        # Calculate attractive force
        att_unit, att_magnitude = self.attractive_force(position, goal)
        attractive_force = att_unit * att_magnitude
        
        # Calculate repulsive force
        rep_unit, rep_magnitude = self.repulsive_force(position, obstacles)
        repulsive_force = rep_unit * rep_magnitude
        
        # Combine forces
        total_force = attractive_force + repulsive_force
        total_magnitude = np.linalg.norm(total_force)
        
        # Prepare force information for debugging/display
        force_info = {
            'attractive_unit': att_unit,
            'attractive_magnitude': att_magnitude,
            'attractive_force': attractive_force,
            'repulsive_unit': rep_unit,
            'repulsive_magnitude': rep_magnitude,
            'repulsive_force': repulsive_force,
            'total_force': total_force,
            'total_magnitude': total_magnitude
        }
        
        return total_force, total_magnitude, force_info


# Global parameters and objects
k_att = 10.0  # Attractive force coefficient
k_rep = 20.0  # Repulsive force coefficient
d0 = 200.0  # Obstacle influence range
dt = 1  # Time step
max_att_force = 50.0
max_rep_force = 50.0

# Initialize managers and calculators
obstacle_manager = ObstacleManager(image_shape=(1080, 1920))
apf_calculator = APFCalculator(k_att, k_rep, d0, max_att_force, max_rep_force)

# Create test obstacles that will interact with the window
# Create obstacles near but not directly on the anchor point
test_obstacle_1 = obstacle_manager.create_test_obstacle_mask(
    center_x=650, center_y=650, width=200, height=150  # Moved away from anchor
)
obstacle_manager.add_obstacle_from_mask(test_obstacle_1)

# Create another obstacle for testing
test_obstacle_2 = obstacle_manager.create_test_obstacle_mask(
    center_x=450, center_y=750, width=120, height=80  # Moved away from anchor
)
obstacle_manager.add_obstacle_from_mask(test_obstacle_2)

# Anchor point and robot initialization
anchor_point = np.array([500.0, 700.0])  # Fixed anchor point (user's preferred position)
robot_position = anchor_point.copy()  # Window starts at anchor point
robot_velocity = np.array([0.0, 0.0])
goal = anchor_point.copy()  # Goal is initially the anchor point

# Window parameters (now configurable)
window_width = 200
window_height = 100

# Path tracking
path_data = [robot_position.copy()]

# Visualization setup
x_limit = (-5, 1920)
y_limit = (-5, 1080)
x_range = np.linspace(x_limit[0], x_limit[1], 100)
y_range = np.linspace(y_limit[0], y_limit[1], 100)
X, Y = np.meshgrid(x_range, y_range)


def compute_potential_field_for_visualization(goal: np.ndarray, obstacles: np.ndarray, 
                                            k_att: float, k_rep: float, d0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute potential field for visualization purposes.
    
    This function is separate from force calculations to avoid confusion
    between visualization needs and actual force computation.
    
    Args:
        goal: Goal position (2,).
        obstacles: Array of obstacle points (N, 2).
        k_att: Attractive force coefficient.
        k_rep: Repulsive force coefficient.
        d0: Obstacle influence range.
        
    Returns:
        Tuple of (potential_field, force_field_x, force_field_y).
    """
    # Attractive potential and force
    diff_goal = np.stack([X - goal[0], Y - goal[1]], axis=-1)
    att_potential = 0.5 * k_att * np.sum(diff_goal**2, axis=-1)
    force_att = -k_att * diff_goal
    
    # Repulsive potential and force
    rep_potential = np.zeros_like(X)
    force_rep = np.zeros(X.shape + (2,))
    
    for obs in obstacles:
        diff_obs = np.stack([X - obs[0], Y - obs[1]], axis=-1)
        dist = np.linalg.norm(diff_obs, axis=-1)
        mask = dist < d0
        
        if np.any(mask):
            safe_dist = np.where(mask, dist, 1)
            rep_potential[mask] += 0.5 * k_rep * ((1/safe_dist[mask]) - (1/d0))**2
            
            coeff = k_rep * ((1/safe_dist[mask]) - (1/d0)) * (1/(safe_dist[mask]**2))
            coeff = coeff[:, np.newaxis]
            direction = diff_obs[mask] / safe_dist[mask][:, np.newaxis]
            rep_force = coeff * direction
            
            # Limit maximum repulsive force
            norm_rep = np.linalg.norm(rep_force, axis=-1)
            over = norm_rep > max_rep_force
            if np.any(over):
                rep_force[over] = rep_force[over] / norm_rep[over][:, np.newaxis] * max_rep_force
            
            force_rep[mask] += rep_force
    
    # Combine fields
    Z = att_potential + rep_potential
    U = force_att[..., 0] + force_rep[..., 0]
    V = force_att[..., 1] + force_rep[..., 1]
    
    return Z, U, V


# Initialize visualization
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(x_limit)
ax.set_ylim(y_limit)

# Plot elements
anchor_plot, = ax.plot(anchor_point[0], anchor_point[1], 'mo', markersize=12, label='Anchor Point')
goal_plot, = ax.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')
obstacles_plot, = ax.plot([], [], 'ro', markersize=8, label='Obstacles')
path_plot, = ax.plot([], [], 'b-', linewidth=2, label='Robot Path')
robot_plot, = ax.plot([], [], 'bo', markersize=8, label='Robot')

# Window rectangle
window_rect = Rectangle((robot_position[0] - window_width/2, robot_position[1] - window_height/2),
                       window_width, window_height, linewidth=2, edgecolor='orange', 
                       facecolor='none', label='Window')
ax.add_patch(window_rect)

# Potential field visualization (using actual obstacle centers for visualization)
sample_obstacles = np.array([[650, 650], [450, 750]])  # Actual obstacle centers
Z, U, V = compute_potential_field_for_visualization(goal, sample_obstacles, 
                                                   k_att, k_rep, d0)
im = ax.imshow(Z, extent=[x_limit[0], x_limit[1], y_limit[0], y_limit[1]], 
               origin='lower', cmap='viridis', alpha=0.6, aspect='auto')

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Potential Value', fontsize=12)

# Force field arrows
quiver_all = ax.quiver(X, Y, U, V, color='white', alpha=0.5)
quiver_robot = ax.quiver([], [], [], [], color='red')

# Force information text
force_text = ax.text(0.65, 0.98, '', transform=ax.transAxes, fontsize=12, 
                     verticalalignment='top', color='black', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


def init():
    """Initialize animation."""
    path_plot.set_data([], [])
    robot_plot.set_data([], [])
    return path_plot, robot_plot


def update(frame):
    """Update animation frame."""
    global robot_position, robot_velocity, goal, quiver_all, quiver_robot, im, window_rect, force_text
    
    # Get current window bounding box
    window_bbox = [robot_position[0] - window_width/2, robot_position[1] - window_height/2,
                   robot_position[0] + window_width/2, robot_position[1] + window_height/2]
    
    # Get obstacle points in window vicinity
    obstacles = obstacle_manager.get_obstacle_points_in_window(window_bbox)
    
    # Dynamic goal selection based on obstacle presence
    if len(obstacles) > 0:
        # If obstacles detected, find the minimum potential point
        # This will be the optimal position that balances avoiding obstacles and staying near anchor
        goal = apf_calculator.find_potential_minimum(anchor_point, obstacles)
    else:
        # If no obstacles, goal is the anchor point
        goal = anchor_point.copy()
    
    # Calculate total force using new APF calculator
    total_force, total_magnitude, force_info = apf_calculator.total_force(
        robot_position, goal, obstacles
    )
    
    # Update robot position - only move if there's a significant force
    if total_magnitude > 0.1:  # Threshold to prevent micro-movements
        robot_velocity += total_force
        # Normalize velocity to prevent excessive speed
        velocity_magnitude = np.linalg.norm(robot_velocity)
        if velocity_magnitude > 5.0:  # Maximum speed limit
            robot_velocity = robot_velocity / velocity_magnitude * 5.0
        robot_position = robot_position + robot_velocity * dt
    else:
        # If no significant force, gradually reduce velocity
        robot_velocity *= 0.9
    
    path_data.append(robot_position.copy())
    
    # Update visualization
    if len(obstacles) > 0:
        obstacles_plot.set_data(obstacles[:, 0], obstacles[:, 1])
    else:
        obstacles_plot.set_data([], [])
    
    # Update path and robot position
    path = np.array(path_data)
    path_plot.set_data(path[:, 0], path[:, 1])
    robot_plot.set_data([robot_position[0]], [robot_position[1]])
    
    # Update goal position visualization
    goal_plot.set_data([goal[0]], [goal[1]])
    
    # Update window rectangle
    window_rect.set_xy((robot_position[0] - window_width/2, robot_position[1] - window_height/2))
    
    # Update potential field visualization based on current obstacles
    if len(obstacles) > 0:
        # Use actual obstacle points for visualization (sample for performance)
        sample_points = obstacles[::10] if len(obstacles) > 10 else obstacles
        Z, U, V = compute_potential_field_for_visualization(goal, sample_points, 
                                                           k_att, k_rep, d0)
        im.set_data(Z)
        quiver_all.set_UVC(U, V)
    
    # Update force visualization
    quiver_robot.remove()
    quiver_robot = ax.quiver(robot_position[0], robot_position[1], 
                            total_force[0], total_force[1], 
                            color='red', scale=1, width=0.005)
    
    # Update force information text
    force_text.set_text(
        f"Anchor: [{anchor_point[0]:.0f}, {anchor_point[1]:.0f}]\n"
        f"Goal: [{goal[0]:.0f}, {goal[1]:.0f}]\n"
        f"Obstacles: {len(obstacles)} points\n"
        f"Attractive: [{force_info['attractive_unit'][0]:.2f}, {force_info['attractive_unit'][1]:.2f}] "
        f"mag={force_info['attractive_magnitude']:.2f}\n"
        f"Repulsive: [{force_info['repulsive_unit'][0]:.2f}, {force_info['repulsive_unit'][1]:.2f}] "
        f"mag={force_info['repulsive_magnitude']:.2f}\n"
        f"Total: [{total_force[0]:.2f}, {total_force[1]:.2f}] mag={total_magnitude:.2f}\n"
        f"Velocity: [{robot_velocity[0]:.2f}, {robot_velocity[1]:.2f}]"
    )
    
    # Check goal reached
    if np.linalg.norm(robot_position - goal) < 0.5:
        print(f"Goal reached at frame {frame}!")
    
    return path_plot, robot_plot, obstacles_plot, anchor_plot, im, quiver_all, window_rect, force_text


# Create animation
ani = animation.FuncAnimation(fig, update, frames=2000, init_func=init, 
                             blit=False, interval=50, repeat=False)

# Setup plot
ax.legend(loc='upper left')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Fixed APF Path Planning with Test Obstacles')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show() 