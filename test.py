results = [1, 4294938646, 1, 1, 1, 1, 1, 1]

# 使用列表推导式找到所有不等于 1 的 ID
special_ids = [x for x in results if x != 1][0]

print(special_ids)