import tkinter as tk
from tkinter import filedialog, messagebox
import base64
from PIL import Image, ImageTk
import io

def convert_image_to_base64():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if not file_path:
        return
    
    try:
        width = int(width_entry.get())
        height = int(height_entry.get())
        
        with open(file_path, "rb") as image_file:
            image = Image.open(image_file)
            image = image.resize((width, height), Image.ANTIALIAS)  # Resize image
            
            image_data = io.BytesIO()
            image.save(image_data, format="PNG")
            image_data = image_data.getvalue()
            
            base64_bytes = base64.b64encode(image_data)
            base64_string = base64_bytes.decode('utf-8')

            base64_text.delete("1.0", tk.END)
            base64_text.insert(tk.END, base64_string)
            save_button.config(state=tk.NORMAL)
            
            # Display the resized image in a label for preview
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail((300, 300))  # Resize for display purposes
            img_tk = ImageTk.PhotoImage(image)
            image_label.config(image=img_tk)
            image_label.image = img_tk
            
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert image: {e}")

def save_base64_text():
    base64_string = base64_text.get("1.0", tk.END).strip()
    if not base64_string:
        messagebox.showwarning("Warning", "No base64 text to save.")
        return
    
    try:
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", 
                                                 filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as text_file:
                text_file.write(base64_string)
            messagebox.showinfo("Success", f"Base64 text saved to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save base64 text: {e}")

# Create the main application window
app = tk.Tk()
app.title("Image to Base64 Converter")

# Frame for image preview and base64 text
frame_top = tk.Frame(app)
frame_top.pack(padx=10, pady=10)

# Entry fields for width and height
width_label = tk.Label(frame_top, text="Width:")
width_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
width_entry = tk.Entry(frame_top, width=10)
width_entry.grid(row=0, column=1, padx=5, pady=5)
width_entry.insert(tk.END, "100")  # Default width

height_label = tk.Label(frame_top, text="Height:")
height_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
height_entry = tk.Entry(frame_top, width=10)
height_entry.grid(row=1, column=1, padx=5, pady=5)
height_entry.insert(tk.END, "100")  # Default height

# Button to select image file
convert_button = tk.Button(frame_top, text="Select Image", command=convert_image_to_base64)
convert_button.grid(row=2, column=0, columnspan=2, pady=10)

# Label for displaying the image preview
image_label = tk.Label(frame_top)
image_label.grid(row=3, column=0, columnspan=2, pady=10)

# Text box for displaying base64 text
base64_text = tk.Text(app, height=10, width=50)
base64_text.pack(pady=10)

# Button to save base64 text to a file
save_button = tk.Button(app, text="Save Base64 Text", command=save_base64_text, state=tk.DISABLED)
save_button.pack(pady=5)

# Run the tkinter main loop
app.mainloop()