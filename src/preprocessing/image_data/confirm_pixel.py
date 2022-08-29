import tkinter as tk
import tkinter.filedialog as tkfd
import cv2
import matplotlib.pyplot as plt
import pandas as pd

root = tk.Tk()
root.withdraw()

# select jpg file
jpg_path = tkfd.askopenfilename(filetypes=[("JPG", "*.jpg")])
img = cv2.imread(jpg_path)

# show pixel info as dataframe
b, g, r = cv2.split(img)
b_df = pd.DataFrame(b)
print(b_df.shape)
print(b_df.head())

# gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
plt.imshow(gray_img, cmap="gray")

# binary image
ret, bin_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
plt.imshow(bin_img, cmap="gray")
bin_df = pd.DataFrame(bin_img)
print(bin_df.shape)
print(bin_df.head())

plt.show()
