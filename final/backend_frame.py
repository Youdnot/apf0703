# Progres
# pv done
# detection done
# force calculation
# ui control

from core.pv_stream import *
from core.detection import *
from core.calculate_force import *
from core.ui_control import *
from utils.mask_utils import *
from utils.convert_coordinate import *
from utils.thread_utils import *
from config import *

#------------------------------------------------------------------------------

# Parameter settings
config_manager = ConfigManager()
hololens_config = config_manager.hololens_config
host = hololens_config.host

detection_config = config_manager.detection_config
output_dir = detection_config.output_dir
prompt_text = detection_config.init_prompt_text
detection_interval = detection_config.detection_interval
# max_frames = 300  # Maximum number of frames to process (prevents infinite loop)

# Movement settings
view_config = config_manager.view_config
window_config = config_manager.window_config
physics_config = config_manager.physics_config
sim_config = config_manager.sim_config

# # 初始化路径数据
# path_data = [sim_config.init_pos.copy()]

# 当前位置和速度
cur_pos = sim_config.init_pos.copy()
cur_vel = sim_config.init_vel.copy()

converted_pos = np.array([0, 0, 0])

# Initialize obstacle mask
obstacle_mask = np.zeros((view_config.width, view_config.height), dtype=bool)

os.makedirs(output_dir, exist_ok=True)

# Set mirror for huggingface
# Don't seem to work
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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
# Initialize the frame producer

# frame_producer = FrameProducer(
#     stack=frame_stack,
#     host=host,
#     calibration_path=calibration_path,
#     pv_width=pv_width,
#     pv_height=pv_height,
#     pv_fps=pv_fps,
#     max_depth=max_depth,
# )

# p = Process(target=frame_producer.run, name="ProducerProcess")
# p.start()

#------------------------------------------------------------------------------
# Initialize the UI control

# Initialize connection and create element
init()

# create ui element
# there will be 1 or 2 sets of UI

# baseline 1: UI only change content; use interaction as the process
# baseline 2: add alert frame; add another process monitering mask condition to turn on and off

# proposed method: UI change content and position; add another process to monitor the position and change the content position
# be careful about the competition of requests

#------------------------------------------------------------------------------

# Main loop
print("[Info] Camera opened. Press 'q' to quit.")
frame_idx = 0

# 修改主循环中的队列访问
while True:
    # 持续重试直到获取到帧
    while True:
        try:
            # pv_frame = frame_queue.peek()
            rgb, pv_z, timestamp, pose = frame_stack.peek()
            pv_frame = rgb
            break  # 成功获取到帧，跳出内层循环
        except Exception as e:
            print(f"Error in frame consumer: {e}")
            continue
        except frame_queue.empty():
            time.sleep(0.01)  # 短暂等待后重试
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
    merged_bool_mask = get_merged_bool_mask_new(current_mask_dict, pv_z)
    print(f"merged_bool_mask: {merged_bool_mask.shape}")
    # 将布尔掩码转换为uint8格式用于显示
    merged_mask_display = (merged_bool_mask * 255).astype(np.uint8)
    # cv2.imshow("Merged Bool Mask", merged_mask_display)

    # process_image_bgr = cv2.cvtColor(process_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Live Inference", process_image_bgr)
    
    # 添加waitKey以确保窗口更新
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("[Info] Quit signal received.")
        break

    # tracker.save_current_state(output_dir=output_dir, raw_image=pv_frame)
    frame_idx += 1

    # if frame_idx >= max_frames:
    #     print(f"[Info] Reached max_frames {max_frames}. Stopping.")
    #     break

    #------------------------------------------------------------------------------
    # UI control

    # Transpose and flip the maskto match the coordinate system
    obstacle_mask = np.flip(merged_bool_mask.T, axis=(1))

    mask_queue.push(obstacle_mask)

# client.close()
p.join()

disconnect()    # disconnect from the UI stream
listener.join()

hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
cv2.destroyAllWindows()
print("[Done] Live inference complete.")