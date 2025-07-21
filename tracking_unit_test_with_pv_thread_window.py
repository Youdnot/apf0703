from utils.camera_tracking import *
from utils.pv_stream import *
from utils.calculate_force import get_attractive_force, get_repulsive_force, get_total_force, update_position_and_velocity
from utils.convert_coordinate import convert_coordinates
from config import config_manager

#------------------------------------------------------------------------------

# 获取配置
view_config = config_manager.view_config
window_config = config_manager.window_config
physics_config = config_manager.physics_config
sim_config = config_manager.sim_config
hololens_config = config_manager.hololens_config

# host = '169.254.10.1'
host = hololens_config.host

# 初始化路径数据
path_data = [sim_config.init_pos.copy()]

# 当前位置和速度
cur_pos = sim_config.init_pos.copy()
cur_vel = sim_config.init_vel.copy()

converted_pos = np.array([0, 0, 0])

# Initialize obstacle mask
obstacle_mask = np.zeros((view_config.width, view_config.height), dtype=bool)
obstacle_mask[500:600, 600:700] = True

cur_pos[0] -= 100
cur_pos[1] -= 50


#------------------------------------------------------------------------------


# Parameter settings
output_dir = "./outputs"
prompt_text = "hand."
detection_interval = 20
# max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

os.makedirs(output_dir, exist_ok=True)

# Initialize the object tracker
tracker = IncrementalObjectTracker(
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_large.pt",
    device="cuda",
    prompt_text=prompt_text,
    detection_interval=detection_interval,
)
tracker.set_prompt("obstacle. person. viehicle. car. bus. truck. desk. table. chair.")

# PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

listener = hl2ss_utilities.key_listener(keyboard.Key.esc)
listener.open()

client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
client.open()

# Open the camera (or replace with local video file, e.g., cv2.VideoCapture("video.mp4"))
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("assets/walking test data.mp4")
# if not cap.isOpened():
#     print("[Error] Cannot open camera.")
#     return

print("[Info] Camera opened. Press 'q' to quit.")
frame_idx = 0

stop_event = threading.Event()
producer = threading.Thread(target=frame_producer, args=(client, stop_event))
producer.start()

try:
    while True:
        # ret, frame = cap.read()
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01) # Wait a tiny bit if no frame is ready
            continue

        cv2.imshow('Video', frame)
        # if not frame:
        #     print("[Warning] Failed to capture frame.")
        #     break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"[Frame {frame_idx}] Processing live frame...")
        process_image = tracker.add_image(frame_rgb)

        if process_image is None or not isinstance(process_image, np.ndarray):
            print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
            frame_idx += 1
            continue

        # 从tracker获取当前mask和元数据
        current_mask_dict = tracker.last_mask_dict
        
        # 合并所有mask为bool数组
        merged_bool_mask = get_merged_bool_mask(current_mask_dict)
        obstacle_mask = merged_bool_mask

        # 将布尔掩码转换为uint8格式用于显示
        merged_mask_display = (merged_bool_mask * 255).astype(np.uint8)
        cv2.imshow("Merged Bool Mask", merged_mask_display)
        
        # 打印基本信息
        # print(f"[Frame {frame_idx}] 合并mask形状: {merged_bool_mask.shape}")
        # print(f"[Frame {frame_idx}] 检测到的对象像素数: {np.sum(merged_bool_mask)}")
        # print(f"[Frame {frame_idx}] 检测到的对象数量: {len(current_mask_dict.labels)}")

        force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, sim_config.anchor_point, obstacle_mask.T, view_config.width, view_config.height, physics_config.d0, physics_config.k_att, physics_config.k_rep, physics_config.damping_factor, physics_config.max_v, physics_config.dt, path_data)

        window_cv_plot = np.zeros((view_config.height, view_config.width, 3), dtype=np.uint8)
        cv2.rectangle(window_cv_plot, (cur_pos[0]-100, view_config.height-(cur_pos[1]+100)), 
                        (cur_pos[0]+100, view_config.height - (cur_pos[1]-100)), (0, 0, 255), 2)
        cv2.imshow("window_cv_plot", window_cv_plot)
        
        # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Live Inference", process_image_bgr)
        
        # 添加waitKey以确保窗口更新
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Info] Quit signal received.")
            break

        tracker.save_current_state(output_dir=output_dir, raw_image=frame_rgb)
        frame_idx += 1

        # if frame_idx >= max_frames:
        #     print(f"[Info] Reached max_frames {max_frames}. Stopping.")
        #     break
except KeyboardInterrupt:
    print("[Info] Interrupted by user (Ctrl+C).")
finally:
    # cap.release()
    cv2.destroyAllWindows()
    print("[Done] Live inference complete.")
