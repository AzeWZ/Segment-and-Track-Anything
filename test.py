import os
import cv2
from model_args import segtracker_args,sam_args,aot_args
from PIL import Image
import numpy as np
import torch
import gc
import imageio
from scipy.ndimage import binary_dilation


import cv2
import cv2

# 读取视频文件和掩码文件
cap = cv2.VideoCapture('b.mp4')
# 读取 masks 文件夹
mask_path = 'masks'
mask_list = os.listdir(mask_path)
# mask_list 排序
print(mask_list)

# 读取视频中每一帧，与 masks 中的掩码文件进行按位与操作
frame_idx =0 
while True:
    ret, frame = cap.read()
    if ret:
        mask = cv2.imread(os.path.join(mask_path,f"{str(frame_idx).zfill(5)}.png"))
        frame = cv2.bitwise_and(frame, mask)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    frame_idx = frame_idx+1
cap.release()

