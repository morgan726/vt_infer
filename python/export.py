from ultralytics import YOLO
model = YOLO("runs/weights/best.pt")
success = model.export(format="onnx", opset=13)