from ultralytics import YOLO

model = YOLO('yolov8n.yaml')

results = model.train(data='cnw.yaml', epochs=1, patience=5)

path = model.export()
print(path)
val = model.val()


'''
optimizer: AdamW(lr=0.001667, momentum=0.9)
 with parameter groups 57 weight(decay=0.0),
  64 weight(decay=0.0005), 
  63 bias(decay=0.0)
Image sizes 640 train, 640 val
'''
