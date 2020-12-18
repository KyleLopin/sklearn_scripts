# Copyright (c) 2019 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitatus Lopin"


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import numpy as np
from skimage import color
from skimage import io
from sklearn.cluster import KMeans


fish = io.imread('fish1.jpg')

print(fish)
print(fish.shape)

red_channel = fish[:, :, 0]
print(red_channel)
print(type(red_channel))

red_column = red_channel.reshape(-1, 1)
print(red_column)
print(red_column.shape)
green_column = fish[:, :, 1].reshape(-1, 1)
blue_column = fish[:, :, 2].reshape(-1, 1)

rgb_data = np.append(red_column, green_column, axis=1)
rgb_data = np.append(rgb_data, blue_column, axis=1)
print(rgb_data)
print(rgb_data.shape)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(fish[1::10, 1::10, 0],
#            fish[1::10, 1::10, 1], fish[1::10, 1::10, 2])
# ax.scatter(red_column[::100], green_column[::100], blue_column[::100])

print(np.unique(fish, return_counts=True))


kmeans = KMeans(n_clusters=3)
kmeans.fit(rgb_data)

mask = kmeans.labels_ != 0
mask = mask.reshape(fish[:, :, 0].shape)

plt.imshow(mask, cmap=plt.cm.gray)
# plt.imshow(fish[:, :, 2], cmap=plt.cm.gray)
print(kmeans.labels_)
print(type(kmeans.labels_))
print(np.unique(kmeans.labels_, return_counts=True))
plt.show()
