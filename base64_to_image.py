import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import base64
import io

def convert_base64_to_image():
    base64_text = text_box.get("1.0", tk.END).strip()
    try:
        image_data = base64.b64decode(base64_text)
        image = Image.open(io.BytesIO(image_data))
        image.thumbnail((300, 300))  # Resize for display purposes

        img_tk = ImageTk.PhotoImage(image)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        save_button.config(state=tk.NORMAL)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert image: {e}")

def save_image():
    base64_text = text_box.get("1.0", tk.END).strip()
    try:
        image_data = base64.b64decode(base64_text)
        image = Image.open(io.BytesIO(image_data))

        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if file_path:
            image.save(file_path)
            messagebox.showinfo("Success", f"Image saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save image: {e}")

app = tk.Tk()
app.title("Base64 to Image Converter")

text_box = tk.Text(app, height=10, width=50)
text_box.pack(pady=10)

convert_button = tk.Button(app, text="Convert to Image", command=convert_base64_to_image)
convert_button.pack(pady=5)

image_label = tk.Label(app)
image_label.pack(pady=10)

save_button = tk.Button(app, text="Save Image", command=save_image, state=tk.DISABLED)
save_button.pack(pady=5)

app.mainloop()