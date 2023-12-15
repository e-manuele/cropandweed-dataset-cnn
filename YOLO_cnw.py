import cv2
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
results = model.train(data='cropandweed.yaml', epochs=3)
print("FINE")
path = model.export(format='onnx')
model.save("yolov8_cnw.pt")
eval = model.val()
test = model('data/images/vwg-1361-0009.jpg')
image = test.masks


best_model_path = "runs/detect/train/weights/best.pt"
# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
