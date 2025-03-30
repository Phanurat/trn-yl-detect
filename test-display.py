import cv2
import numpy as np
import mss
import pytesseract
from ultralytics import YOLO

model = YOLO('train/runs/train/exp/weights/best.onnx')  # หรือ path ที่เก็บโมเดลของคุณ
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# เริ่มจับภาพหน้าจอแบบเรียลไทม์
with mss.mss() as sct:
    while True:
        # จับภาพหน้าจอ
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        # แปลงจาก BGRA เป็น BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # 🔹 **1) ตรวจจับวัตถุด้วยโมเดล YOLO**
        results = model(img)
        result = results[0]  # ใช้ผลลัพธ์จากการทำนายแรก (กรณีทำนายหลายภาพ)

        # วาดกรอบและแสดงผลการตรวจจับวัตถุ
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดของ bounding box
            conf = box.conf[0]  # ความมั่นใจของการตรวจจับ
            label = result.names[int(box.cls[0])]  # ประเภทของวัตถุที่ตรวจจับได้

            # วาดกรอบรอบวัตถุ
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 🔹 **2) ตรวจจับข้อความด้วย Tesseract OCR**
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(img_gray, lang="eng", config="--psm 6")  # อ่านข้อความจากภาพ
        text = text.strip()

        # แสดงข้อความที่ตรวจจับได้
        if text:
            print("Detected Text:", text)
            cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # แสดงภาพที่มีการตรวจจับวัตถุและข้อความ
        cv2.imshow("YOLOv8 + OCR Detection", img)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดหน้าต่างทั้งหมด
cv2.destroyAllWindows()
