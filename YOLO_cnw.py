from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='cnw.yaml', epochs=5)

path = model.export(format='onnx')
model.save("yolov8n_cnw.pt")
val = model.val()

'''

optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.001667, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 640 train, 640 val

'''