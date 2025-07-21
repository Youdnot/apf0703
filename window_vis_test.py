import numpy as np
import cv2

cur_pos = np.array([500, 700])

window_cv_plot = np.zeros((1080, 1980, 3), dtype=np.uint8)
cv2.rectangle(window_cv_plot, (cur_pos[0]-100, cur_pos[1]+100), (cur_pos[0] + 100, cur_pos[1] - 100), (0, 0, 255), 2)


cv2.imshow("window_cv_plot", window_cv_plot)
cv2.waitKey(10000)

cv2.destroyAllWindows()