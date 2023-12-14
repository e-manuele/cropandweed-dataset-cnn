import os

from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')
results = model.train(data='cropandweed.yaml', epochs=1)
print("FINE")
eval = model.val()
test = model('data/images/vwg-1361-0009.jpg')
os.system("cnw/visualize_annotations.py --dataset CropsOrWeed2 --filter vwg-1361-0009")
success = model.export(format='onnx')

print('eval')
print(eval)
print('test')
print(test)
print('success')
print(success)
