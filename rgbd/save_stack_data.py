import os
import cv2
import numpy as np
import csv

# Directory to save the data
save_dir = './saved_data'
os.makedirs(save_dir, exist_ok=True)

# Function to save RGB and Depth frames along with metadata
def save_data(rgb, pv_z, timestamp, pose, index):
    # Save RGB image
    rgb_filename = os.path.join(save_dir, f'rgb_{index}.png')
    cv2.imwrite(rgb_filename, rgb)

    # Save Depth map
    depth_filename = os.path.join(save_dir, f'depth_{index}.npy')
    np.save(depth_filename, pv_z)

    # Save metadata
    metadata_filename = os.path.join(save_dir, 'metadata.csv')
    with open(metadata_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([index, timestamp, pose])

# Example usage
if __name__ == "__main__":
    # Assuming frame_stack is available and populated
    index = 0
    while not frame_stack.empty():
        rgb, pv_z, timestamp, pose = frame_stack.pop()
        save_data(rgb, pv_z, timestamp, pose, index)
        index += 1

