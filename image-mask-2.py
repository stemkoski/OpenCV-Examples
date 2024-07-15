import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

# Function to load and display the image
def load_image():
    global img, hsv_img, img_display
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.resize( img, (512, 512) )
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        apply_hsv_threshold()

# Function to display the image in the Tkinter window
def display_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)
    img_display.config(image=img_tk)
    img_display.image = img_tk

# Function to apply the HSV threshold and replace selected pixels with magenta
def apply_hsv_threshold(*args):
    global img, hsv_img, img_display
    if img is None:
        return
    
    lower_h = lower_hue.get()
    lower_s = lower_saturation.get()
    lower_v = lower_value.get()
    upper_h = upper_hue.get()
    upper_s = upper_saturation.get()
    upper_v = upper_value.get()

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    img_copy = img.copy()
    img_copy[mask > 0] = [255, 0, 255]  # Magenta color in BGR

    display_image(img_copy)

# Create the main window
root = tk.Tk()
root.title("Image HSV Thresholding")

# Create and place the image display label
img_display = tk.Label(root)
img_display.grid(row=0, column=0, columnspan=6)

# Create and place the load image button
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.grid(row=1, column=0, columnspan=6)

# HSV threshold sliders and labels
lower_hue = tk.Scale(root, from_=0, to=179, orient="horizontal", label="Lower Hue", command=apply_hsv_threshold)
lower_hue.grid(row=2, column=0)
upper_hue = tk.Scale(root, from_=0, to=179, orient="horizontal", label="Upper Hue", command=apply_hsv_threshold)
upper_hue.grid(row=2, column=1)
lower_saturation = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Lower Saturation", command=apply_hsv_threshold)
lower_saturation.grid(row=2, column=2)
upper_saturation = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Upper Saturation", command=apply_hsv_threshold)
upper_saturation.grid(row=2, column=3)
lower_value = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Lower Value", command=apply_hsv_threshold)
lower_value.grid(row=2, column=4)
upper_value = tk.Scale(root, from_=0, to=255, orient="horizontal", label="Upper Value", command=apply_hsv_threshold)
upper_value.grid(row=2, column=5)

# Initialize the image and hsv_img variables
img = None
hsv_img = None

# Start the main event loop
root.mainloop()