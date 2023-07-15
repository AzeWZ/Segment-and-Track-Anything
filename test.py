import os
import cv2
import numpy as np
from PIL import Image

# 读取视频文件和掩码文件
cap = cv2.VideoCapture('b.mp4')
# 读取 masks 文件夹
mask_path = 'masks'
mask_list = os.listdir(mask_path)
# mask_list 排序
# print(mask_list)

# 读取视频中每一帧，与 masks 中的掩码文件进行按位与操作
frame_idx = 0

while True:
    ret, frame = cap.read()
    if ret:
        mask = cv2.imread(os.path.join(mask_path, f"{str(frame_idx).zfill(5)}.png"))
        # mask二值化
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        print(mask.shape)
        # 读取的 mask 是三通道的，需要转换成四通道的
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
        # 保存 mask 黑白两色
        # cv2.imwrite(f"{str(frame_idx).zfill(5)}_c.png", mask[:, :, ::1])
        # 融合 frame 和 mask

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)

        frame = cv2.bitwise_and(frame, mask)
        print(frame.shape)
        print(frame[0, 0])
        # Slice of alpha channel
        alpha = frame[:, :, 3]

        # Use logical indexing to set alpha channel to 0 where BGR=0
        alpha[np.all(frame[:, :, 0:3] == (0, 0, 0), 2)] = 0
        print(frame[0, 0])
        # 保存融合后的图片
        cv2.imwrite(f"{str(frame_idx).zfill(5)}_b.png", frame)
        # cv2.imwrite(f"{str(frame_idx).zfill(5)}_b.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    else:
        break
    frame_idx = frame_idx + 1
cap.release()
# END: 3j5d9f8d4j3d

