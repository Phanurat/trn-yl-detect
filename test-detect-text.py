from ultralytics import YOLO
import cv2
import os
import pytesseract

# โหลดโมเดล
model = YOLO('train/runs/train/exp2/weights/best.onnx')

# กำหนด path ของ Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# อ่านภาพจากไฟล์
img_path = 'n-img1.png'
img = cv2.imread(img_path)

# ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
if img is None:
    print(f"ไม่พบไฟล์รูปภาพ: {img_path}")
    exit()

# ปรับขนาดภาพ (ถ้าภาพมีขนาดเล็ก)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# ทำการทำนาย
results = model(img)

# ดึงผลลัพธ์จากลิสต์
result = results[0]

# ตั้งค่าความแม่นยำขั้นต่ำ
confidence_threshold = 0.79

# โฟลเดอร์สำหรับบันทึกภาพที่ถูกตัด
output_folder = 'cropped_objects'
os.makedirs(output_folder, exist_ok=True)

# ดึง Bounding Boxes และ Confidence Score
for i, (box, conf) in enumerate(zip(result.boxes.xyxy, result.boxes.conf)):
    if conf >= confidence_threshold:  # ตรวจสอบความแม่นยำ
        x1, y1, x2, y2 = map(int, box)  # แปลงค่าเป็น int

        # ตรวจสอบว่าพิกัด Bounding Box อยู่ในขอบเขตของภาพ
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped_img = img[y1:y2, x1:x2]  # ตัดภาพตาม Bounding Box

        # ตรวจสอบว่า cropped_img มีข้อมูลหรือไม่
        if cropped_img.size > 0:
            # แปลงภาพเป็นขาวดำเพื่อการตรวจจับข้อความที่ดีขึ้น
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            # ใช้ Thresholding เพื่อลดการแยกคำที่ไม่จำเป็น
            _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # ใช้ Tesseract OCR ตรวจจับข้อความทั้งภาษาไทยและภาษาอังกฤษ
            text = pytesseract.image_to_string(thresh_img, lang="tha+eng")  # ใช้ภาษาไทยและภาษาอังกฤษ

            # แสดงผลลัพธ์
            print("ข้อความที่ตรวจพบ:")
            print(text)

            # บันทึกภาพที่ตัด
            filename = os.path.join(output_folder, f'cropped_object_{i}.png')
            cv2.imwrite(filename, cropped_img)
            print(f"บันทึก {filename} (Confidence: {conf:.2f})")
        else:
            print(f"ภาพที่ตัดออกมาไม่สามารถใช้ OCR ได้ (cropped_object_{i})")

# แสดงภาพที่ตรวจจับ
result.show()
