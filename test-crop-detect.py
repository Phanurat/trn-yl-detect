from ultralytics import YOLO
import cv2
import os
import pytesseract

# โหลดโมเดล
model = YOLO('train/runs/train/exp2/weights/best.onnx')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# อ่านภาพจากไฟล์
img_path = 'n-img1.png'
img = cv2.imread(img_path)

# ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
if img is None:
    print(f"ไม่พบไฟล์รูปภาพ: {img_path}")
    exit()

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
        image_path = cropped_img
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang="tha")  # ใช้แค่ภาษาไทย

        # แสดงผลลัพธ์
        print("ข้อความภาษาไทยที่ตรวจพบ:")
        print(text)

        # ตรวจสอบว่า cropped_img ไม่ว่างเปล่า
        if cropped_img.size > 0:
            filename = os.path.join(output_folder, f'cropped_object_{i}.png')
            cv2.imwrite(filename, cropped_img)
            print(f"บันทึก {filename} (Confidence: {conf:.2f})")

# แสดงภาพที่ตรวจจับ
result.show()
