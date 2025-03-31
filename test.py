from ultralytics import YOLO
import cv2

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = YOLO('train/runs/train/exp3/weights/best.onnx')  # หรือ path ที่เก็บโมเดลของคุณ

# อ่านภาพจากไฟล์
img = cv2.imread('3103test002.png')  # ใส่ path ของภาพที่ต้องการทำนาย

# ทำการทำนาย
results = model(img)

# ดึงผลลัพธ์จากลิสต์ (กรณีที่ทำนายหลายๆ ภาพ)
result = results[0]

# แสดงผลลัพธ์
result.show()  # แสดงภาพที่ทำนาย
result.save()  # บันทึกผลลัพธ์ (ถ้าต้องการบันทึกภาพที่ทำนายแล้ว)
