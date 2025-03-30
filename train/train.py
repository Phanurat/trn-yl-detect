from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# ตั้งค่าไฟล์ข้อมูล
DATA_YAML = "../data.yaml"

# ฝึกโมเดล
model.train(
    data=DATA_YAML,  
    epochs=50,       
    batch=8,          
    device='cpu',     
    workers=8,        
    project="runs/train",  
    name="exp"        
)

# บันทึกโมเดลหลังฝึก
model.export(format="onnx")  
