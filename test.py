from datetime import datetime

# 获取当前日期时间，格式化为字符串，精确到分钟
current_datetime = datetime.now().strftime("%Y-%m-%d-%H:%M")

print(current_datetime)

print("outputs/" + datetime.now().strftime("%Y-%m-%d-%H%M"))