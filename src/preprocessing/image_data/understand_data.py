import tkinter as tk
import tkinter.filedialog as tkfd
import cv2
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

# select jpg file
jpg_path = tkfd.askopenfilename(filetypes=[("JPG", "*.jpg")])
img = cv2.imread(jpg_path)

print(img.shape)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

print(img)

print(len(img))
print(len(img[0]))
print(len(img[0][0]))

plt.show()
