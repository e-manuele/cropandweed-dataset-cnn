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

# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
#
# import cv2
#
# image = test['image']
# bounding_boxes = test['bounding_boxes']
# labels = test['labels']
# print(labels)
# for bounding_box, label in zip(bounding_boxes, labels):
#     cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 255), 2)
#     cv2.putText(image, label, (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
# cv2.imshow('Image', image)
# cv2.waitKey(0)


################
'''

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['im1.jpg', 'im2.jpg'])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
'''