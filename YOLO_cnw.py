from ultralytics import YOLO

models = ['yolov8n.pt']
'''
yolov8n.pt 
yolov8s.pt 
yolov8m.pt 
yolov8l.pt 
yolov8x.pt
'''
for model_item in models:
    model = YOLO(model_item)
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
