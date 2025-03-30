import os
import xml.etree.ElementTree as ET
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

# ฟังก์ชันแปลง XML เป็น YOLO format
def convert_xml_to_yolo(xml_file, image_width, image_height, class_mapping):
    print(f"กำลังเปิดไฟล์ XML: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_data = []

    for member in root.findall('object'):
        class_name = member.find('name').text
        class_id = class_mapping.get(class_name, None)

        if class_id is None:
            print(f"ไม่พบคลาสใน class_mapping สำหรับ {class_name} ในไฟล์ {xml_file}")
            continue

        # คำนวณพิกัด Bounding Box
        xmin = int(member.find('bndbox/xmin').text)
        ymin = int(member.find('bndbox/ymin').text)
        xmax = int(member.find('bndbox/xmax').text)
        ymax = int(member.find('bndbox/ymax').text)

        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # เก็บข้อมูลในรูปแบบ YOLO
        yolo_data.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return yolo_data


def convert_all_xml_to_yolo(xml_folder, image_folder, output_folder, class_mapping):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith(".xml"):
            print(f"กำลังแปลงไฟล์: {xml_file}")

            xml_path = os.path.join(xml_folder, xml_file)
            image_path = os.path.join(image_folder, xml_file.replace(".xml", ".png"))  # เปลี่ยน .xml เป็น .png

            if not os.path.exists(image_path):
                print(f"ไม่พบไฟล์ภาพ {image_path} สำหรับไฟล์ {xml_file}")
                continue

            img = Image.open(image_path)
            image_width, image_height = img.size

            # แปลง XML ไปเป็น YOLO format
            yolo_data = convert_xml_to_yolo(xml_path, image_width, image_height, class_mapping)

            if not yolo_data:
                print(f"ไม่มีข้อมูลในไฟล์ {xml_file} หรือแปลงไม่ได้")
                continue

            # บันทึกข้อมูลที่แปลงแล้วในไฟล์ .txt
            output_file = os.path.join(output_folder, xml_file.replace(".xml", ".txt"))
            with open(output_file, 'w') as f:
                for line in yolo_data:
                    f.write(f"{line}\n")
            print(f"แปลง {xml_file} เสร็จเรียบร้อย")


# ฟังก์ชัน GUI สำหรับเลือกโฟลเดอร์
def select_folder(title):
    folder = filedialog.askdirectory(title=title)
    if not folder:
        messagebox.showerror("Error", "ไม่พบโฟลเดอร์ที่เลือก")
    return folder


# ฟังก์ชันที่ทำงานเมื่อคลิกปุ่ม "เริ่มแปลง"
def start_conversion():
    # เลือกโฟลเดอร์
    xml_folder = select_folder("เลือกโฟลเดอร์ที่เก็บไฟล์ XML")
    if not xml_folder:
        return

    image_folder = select_folder("เลือกโฟลเดอร์ที่เก็บไฟล์ภาพ")
    if not image_folder:
        return

    output_folder = select_folder("เลือกโฟลเดอร์ที่เก็บผลลัพธ์")
    if not output_folder:
        return

    # กำหนดการแมป class และ class ID
    class_mapping = {'topic': 0, 'Other': 1}  # ปรับการแมปให้เหมาะสมกับคลาสของคุณ

    # เรียกใช้ฟังก์ชันแปลงไฟล์
    convert_all_xml_to_yolo(xml_folder, image_folder, output_folder, class_mapping)
    messagebox.showinfo("เสร็จสิ้น", "แปลงไฟล์เสร็จเรียบร้อยแล้ว")


# สร้างหน้าต่าง GUI
root = tk.Tk()
root.title("XML to YOLO Converter")
root.geometry("400x300")

# เพิ่มคำอธิบาย
label = tk.Label(root, text="เลือกโฟลเดอร์เพื่อแปลงไฟล์ XML เป็น YOLO", font=('Arial', 14))
label.pack(pady=20)

# ปุ่มสำหรับเริ่มแปลง
btn_convert = tk.Button(root, text="เริ่มแปลง", font=('Arial', 14), bg="#3498DB", fg="white", command=start_conversion)
btn_convert.pack(pady=20)

# ข้อความอธิบายเพิ่มเติม
info_label = tk.Label(root, text="โปรดเลือกโฟลเดอร์ที่มีไฟล์ XML, ไฟล์ภาพและโฟลเดอร์ผลลัพธ์", font=('Arial', 10), fg="gray")
info_label.pack(pady=10)

# เริ่ม GUI
root.mainloop()
