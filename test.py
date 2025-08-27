# import numpy as np
# from datetime import datetime, timezone

# # 获取当前的UTC时间
# now_utc = datetime.utcnow()

# # 将其转换为 numpy.datetime64 对象
# utc_time = np.datetime64(now_utc, 'ns') # 使用纳秒精度

# print(f"Python datetime object: {now_utc}")
# print(f"NumPy datetime64 object: {utc_time}")

# import os

# # mirror source
# # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# model_id = "IDEA-Research/grounding-dino-tiny"
# local_path = "./local-grounding-dino"

# # 下载模型和处理器
# model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
# processor = AutoProcessor.from_pretrained(model_id)

# # 保存到本地
# model.save_pretrained(local_path)
# processor.save_pretrained(local_path)


from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

device = "cuda"

processor = AutoProcessor.from_pretrained("./local-grounding-dino")
print("processor success")
model = AutoModelForZeroShotObjectDetection.from_pretrained("./local-grounding-dino").to(device)
print("model success")