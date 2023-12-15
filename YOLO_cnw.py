from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='cropandweed.yaml', epochs=5)

path = model.export(format='onnx')
model.save("yolov8n_cnw.pt")
val = model.val()
