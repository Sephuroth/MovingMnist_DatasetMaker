# contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame.
# 10000 个视频，每个视频20帧，每帧宽64  高64
import numpy as np
import cv2
import os

data = np.load('mnist_test_seq.npy')
data = data.transpose(1, 0, 2, 3)
print(data.shape)  # 10000,20,64,64

for index in range(len(data[:, 0, 0, 0])):
    # print(data[index,...].shape)
    count = 0
    for j in data[index, ...]:
        img = np.expand_dims(j, -1)
        print(img.shape)
        filename = 'movingmnist/train/video_%d/frame_%d.jpg' % (index, count)
        count += 1
        savepath = 'movingmnist/train/video_%d' % (index)
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        cv2.imwrite(filename, img)
        # cv2.imshow('img0', img)
        # cv2.waitKey(0)

