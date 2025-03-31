import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import cv2
from PIL import Image, ImageTk

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = YOLO('train/runs/train/exp3/weights/best.onnx')

# ฟังก์ชันในการโหลดภาพ
def load_image():
    filepath = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if filepath:
        try:
            img = cv2.imread(filepath)
            process_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the image: {str(e)}")

# ฟังก์ชันในการประมวลผลภาพ
def process_image(img):
    # ทำนายภาพ
    results = model(img)
    result = results[0]

    # แสดงผลลัพธ์ใน GUI
    result_img = result.plot()  # ใช้ plot เพื่อดึงภาพที่ทำนายแล้ว
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # แปลงภาพจาก BGR เป็น RGB
    global img_pil, img_tk
    img_pil = Image.fromarray(result_img)  # แปลงเป็น PIL Image
    img_tk = ImageTk.PhotoImage(img_pil)  # แปลงเป็น ImageTk ที่สามารถแสดงใน Tkinter ได้

    # อัปเดตภาพใน GUI
    update_image()

# ฟังก์ชันในการอัปเดตการแสดงภาพ
def update_image():
    # ขนาดของหน้าต่างหลัก
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    # ขนาดของภาพต้นฉบับ
    img_width, img_height = img_pil.size

    # คำนวณอัตราส่วนของภาพ
    aspect_ratio = img_width / img_height

    # คำนวณขนาดใหม่ที่รักษาอัตราส่วน (Aspect Ratio)
    if window_width / window_height > aspect_ratio:
        # กรณีที่หน้าจอสูงกว่ากว้าง
        new_width = int(window_height * aspect_ratio)
        new_height = window_height
    else:
        # กรณีที่หน้าจอกว้างกว่าหรือเท่ากับสูง
        new_width = window_width
        new_height = int(window_width / aspect_ratio)

    # ปรับขนาดภาพให้พอดีกับหน้าต่าง
    img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # แปลงภาพเป็น PhotoImage สำหรับแสดงใน Tkinter
    img_tk_resized = ImageTk.PhotoImage(img_resized)

    # อัปเดตภาพใน Label
    panel.config(image=img_tk_resized)
    panel.image = img_tk_resized  # เก็บอ้างอิงไว้เพื่อไม่ให้ถูกเก็บใน garbage collection

# สร้างหน้าต่างหลัก
root = tk.Tk()
root.title("YOLO Object Detection")

# กำหนดขนาดหน้าต่างเริ่มต้น
root.geometry("800x600")
root.config(bg="#2C3E50")

# ล็อกขนาดหน้าต่าง (ไม่สามารถย่อหรือขยายได้)
root.resizable(False, False)

# ฟอนต์และสีที่ใช้ในปุ่ม
button_font = ('Arial', 14, 'bold')
button_bg = '#3498DB'
button_fg = '#ffffff'
button_hover_bg = '#2980B9'

# ฟังก์ชันสำหรับเปลี่ยนสีปุ่มเมื่อ hover
def on_enter(e):
    btn_load.config(bg=button_hover_bg)

def on_leave(e):
    btn_load.config(bg=button_bg)

# สร้างปุ่มเพื่อเลือกภาพ
btn_load = tk.Button(root, text="Load Image", font=button_font, bg=button_bg, fg=button_fg, relief="raised", bd=5, command=load_image)
btn_load.pack(pady=20)

# เพิ่มการเปลี่ยนสีปุ่มเมื่อ hover
btn_load.bind("<Enter>", on_enter)
btn_load.bind("<Leave>", on_leave)

# สร้าง Label เพื่อแสดงภาพ
panel = tk.Label(root, bg="#2C3E50")
panel.pack(padx=10, pady=10, fill="both", expand=True)

# เพิ่มข้อความอธิบาย
label = tk.Label(root, text="Select an image to detect objects", font=('Arial', 18, 'italic'), bg="#2C3E50", fg="#ecf0f1")
label.pack(pady=10)

# เริ่มต้น GUI
root.mainloop()
