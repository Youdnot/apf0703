import numpy as np
from datetime import datetime, timezone

# 获取当前的UTC时间
now_utc = datetime.utcnow()

# 将其转换为 numpy.datetime64 对象
utc_time = np.datetime64(now_utc, 'ns') # 使用纳秒精度

print(f"Python datetime object: {now_utc}")
print(f"NumPy datetime64 object: {utc_time}")