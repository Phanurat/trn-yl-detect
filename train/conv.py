import os
import xml.etree.ElementTree as ET
from PIL import Image

# ฟังก์ชันแปลง XML เป็น YOLO format
def convert_xml_to_yolo(xml_file, image_width, image_height, class_mapping):
    # อ่านไฟล์ XML
    print(f"กำลังเปิดไฟล์ XML: {xml_file}")  # เพิ่มการพิมพ์
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_data = []

    for member in root.findall('object'):
        class_name = member.find('name').text
        class_id = class_mapping.get(class_name, None)

        # ถ้าไม่พบคลาสในการแมป จะข้ามไฟล์นี้
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
            print(f"กำลังแปลงไฟล์: {xml_file}")  # เพิ่มการพิมพ์

            xml_path = os.path.join(xml_folder, xml_file)
            image_path = os.path.join(image_folder, xml_file.replace(".xml", ".png"))  # เปลี่ยน .xml เป็น .png

            # ตรวจสอบว่าไฟล์ภาพมีอยู่หรือไม่
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

# กำหนดการทำงาน
xml_folder = 'D:/project-ml/datasets/datasets'  # โฟลเดอร์ที่เก็บไฟล์ XML
image_folder = 'D:/project-ml/datasets/train/images'  # โฟลเดอร์ที่เก็บไฟล์ภาพ (เช่น .png)
output_folder = 'D:/project-ml/datasets/train/labels'  # โฟลเดอร์ที่จะบันทึกไฟล์ .txt

# กำหนดการแมป class และ class ID
class_mapping = {'topic': 0, 'Other': 1}  # ปรับการแมปให้เหมาะสมกับคลาสของคุณ

# เรียกใช้ฟังก์ชันแปลงไฟล์
convert_all_xml_to_yolo(xml_folder, image_folder, output_folder, class_mapping)
