import tkinter as tk
import tkinter.filedialog as tkfd
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

# define data set array
pixels = []
labels = []

# select directory
selected_dir = tkfd.askdirectory()

# append pixel value and label of each image to data set array
for current_dir, sub_dirs, file_list in os.walk(selected_dir):
    if len(sub_dirs) > 0:
        for i, d in enumerate(sub_dirs):
            sub_dir_path = current_dir + '/' + d
            files = os.listdir(sub_dir_path)
            
            for f in files:
                img = cv2.imread(sub_dir_path + '/' + f, 0)
                img = cv2.resize(img, (128, 128))
                img = np.array(img).flatten().tolist()  # convert 2d array to 1d array
                pixels.append(img)

                labels.append(i)

# convert data set from list to dataframe
pixels_df = pd.DataFrame(pixels)
pixels_df = pixels_df / 255  # normalize pixel between 0 and 1

labels_df = pd.DataFrame(labels)
labels_df = labels_df.rename(columns={0: "label"})

img_set = pd.concat([pixels_df, labels_df], axis=1)
print(img_set.head())

# morphological transformation
# gray scale image
img = cv2.imread("/workspaces/PythonMachineLearning/preprocess_sample_data/chap5/data/data/ants/VietnameseAntMimicSpider.jpg", 0)
ret, bin_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
plt.imshow(bin_img, cmap="gray")

# erosion
kernel = np.ones((3, 3), np.uint8)
img_el = cv2.erode(bin_img, kernel, iterations=1)
plt.imshow(img_el, cmap="gray")

# dilation
img_dl = cv2.dilate(bin_img, kernel, iterations=1)
plt.imshow(img_dl, cmap="gray")

# opening
img_op = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
plt.imshow(img_op, cmap="gray")

# closing
img_cl = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
plt.imshow(img_cl, cmap="gray")

plt.show()
