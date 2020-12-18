# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"

import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage import io

fish = io.imread('fish1.jpg')
# 1 ======>
# plt.imshow(fish)
# plt.show()
grayscale = color.rgb2gray(fish)

fig, ((ax1, ax2),
      (ax3, ax4),
      (ax5, ax6)) = plt.subplots(nrows=3,  ncols=2, figsize=(6, 10))

ax1.imshow(fish)
print(grayscale)
ax2.imshow(grayscale, cmap=plt.cm.gray)
# ax2.imshow(fish[:, :, 0], cmap=plt.cm.gray)

# create the histogram of greyscale
histogram, bin_edges = np.histogram(grayscale, bins=256, range=(0, 1))
ax4.plot(bin_edges[0:-1], histogram)

# create histogram of rgb
colors = ("r", "g", "b")
channel_ids = (0, 1, 2)

for channel_id, color in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        fish[:, :, channel_id], bins=256, range=(0, 256)
    )
    ax3.plot(bin_edges[0:-1], histogram, color=color)

ax3.set_ylim([0, 1200])
ax4.set_ylim([0, 1200])

# Threshold
threshold = .78
# threshold = threshold_otsu(grayscale)
# print(threshold)
fish_th = grayscale > threshold

ax5.imshow(fish_th, cmap=plt.cm.gray)



plt.show()
