from core.pv_stream import *
from core.detection import *
from core.calculate_force import *
from core.ui_control import *
from utils.mask_utils import *
from utils.keyboard_utils import *
from utils.convert_coordinate import *
from utils.thread_utils import *
from config import config_manager

#------------------------------------------------------------------------------

# Parameter settings
hololens_config = config_manager.hololens_config
host = hololens_config.host

detection_config = config_manager.detection_config
output_dir = detection_config.output_dir
prompt_text = detection_config.init_prompt_text
detection_interval = detection_config.detection_interval

# Movement settings
view_config = config_manager.view_config
window_config = config_manager.window_config
physics_config = config_manager.physics_config
sim_config = config_manager.sim_config

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

os.makedirs(output_dir, exist_ok=True)

# Set mirror for huggingface
# Don't seem to work
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#------------------------------------------------------------------------------

# Initialize thread functions
def frame_producer(client, frame_queue: LifoQueue, stop_event):
    while not stop_event.is_set():
        try:
            data = client.get_next_packet()
            frame = data.payload.image
            frame_queue.put(frame)
        except Exception as e:
            print(f"Error in frame producer: {e}")
            time.sleep(0.1)

def movement_consumer(mask_queue, stop_event):
    while not stop_event.is_set():
        obstacle_mask = mask_queue.get()
        force, cur_pos, cur_vel, converted_pos, path_data = update_position_and_velocity(cur_pos, cur_vel, sim_config.anchor_point, obstacle_mask, view_config.width, view_config.height, physics_config.d0, physics_config.k_att, physics_config.k_rep, physics_config.damping_factor, physics_config.max_v, physics_config.dt, path_data)
        update_position(converted_pos)
        time.sleep(0.1)


#------------------------------------------------------------------------------

# Initialize the object tracker
tracker = IncrementalObjectTracker(
    grounding_model_id="IDEA-Research/grounding-dino-tiny",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
    sam2_ckpt_path="./external/grounding_sam2/checkpoints/sam2.1_hiera_tiny.pt",
    device="cuda",
    prompt_text=prompt_text,
    detection_interval=detection_interval,
)
tracker.set_prompt(detection_config.final_prompt_text)

#------------------------------------------------------------------------------

# Initialize the PV stream
hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, enable_mrc=enable_mrc, shared=shared)

pv_client = hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, mode=mode, width=width, height=height, framerate=framerate, profile=profile, bitrate=bitrate, decoded_format=decoded_format)
pv_client.open()

#------------------------------------------------------------------------------

# Initialize the UI control
# Initialize connection and create element
element_key = initialize_connection()

#------------------------------------------------------------------------------


# Initialize the thread
frame_queue = LifoQueue(maxsize=1)  # 线程安全的队列，最多存储一帧数据
mask_queue = LifoQueue(maxsize=1)  # 线程安全的队列，最多存储一帧掩码数据

# Define a stop event for keyboard listener
stop_event = mp.Event()

listener = keyboard.Listener(on_press=lambda key: on_press(stop_event, key))
listener.start()


# Initialize the frame producer
frame_producer = mp.Process(target=frame_producer, args=(pv_client, frame_queue, stop_event))

# Initialize the movement consumer
movement_consumer = mp.Process(target=movement_consumer, args=(mask_queue, stop_event))

#------------------------------------------------------------------------------

# Main process
frame_idx = 0

frame_producer.start()
movement_consumer.start()

while not stop_event.is_set():
    try:
        pv_frame = frame_queue.get()
    except frame_queue.Empty:
        time.sleep(0.01) # Wait a tiny bit if no frame is ready
        continue

    print(f"[Frame {frame_idx}] Processing live frame...")
    process_image = tracker.add_image(pv_frame)

    if process_image is None or not isinstance(process_image, np.ndarray):
        print(f"[Warning] Skipped frame {frame_idx} due to empty result.")
        frame_idx += 1
        continue

    # 从tracker获取当前mask和元数据
    current_mask_dict = tracker.last_mask_dict
    
    # 合并所有mask为bool数组
    merged_bool_mask = get_merged_bool_mask(current_mask_dict)
    print(f"merged_bool_mask: {merged_bool_mask.shape}")
    # 将布尔掩码转换为uint8格式用于显示
    merged_mask_display = (merged_bool_mask * 255).astype(np.uint8)
    cv2.imshow("Merged Bool Mask", merged_mask_display)

    process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Live Inference", process_image_bgr)
    cv2.waitKey(1)

    # Transpose and flip the maskto match the coordinate system
    obstacle_mask = np.flip(merged_bool_mask.T, axis=(1))
    mask_queue.put(obstacle_mask)

    # tracker.save_current_state(output_dir=output_dir, raw_image=pv_frame)
    frame_idx += 1

# Stop and clean up
pv_client.close()
disconnect()    # disconnect from the UI stream
listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
cv2.destroyAllWindows()
print("[Done] Live inference complete.")