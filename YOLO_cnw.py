from ultralytics import YOLO

models = ['runs/detect/train2/weights/best.pt']
'''
yolov8n.pt 
yolov8s.pt 
yolov8m.pt 
yolov8l.pt 
yolov8x.pt
'''
for model_item in models:
    model = YOLO(model_item)
    #results = model.train(data='cnw.yaml', epochs=100, patience=5, name="fisso")
    #path = model.export()
    #print(path)
    val = model.val(name="BEST")

'''
Ultralytics YOLOv8.0.227 🚀 Python-3.9.13 torch-2.1.1+cpu CPU (AMD Ryzen 5 5600X 6-Core Processor)

trainer:
task=detect, 
mode=train, model=yolov8n.yaml, 
data=cnw.yaml, epochs=100, patience=5, 
batch=16, imgsz=640, save=True, save_period=-1, 
cache=False, device=None, workers=8, project=None, 
name=fisso, exist_ok=False, pretrained=True, optimizer=auto, 
verbose=True, seed=0, deterministic=True, single_cls=False, 
rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, 
fraction=1.0, profile=False, freeze=None, overlap_mask=True, 
mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, 
save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, 
lots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, 
augment=False, agnostic_nms=False, classes=None, retina_masks=False, show=False, 
save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, 
show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, 
optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, 
nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, 
warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, 
label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, 
scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, 
copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs\detect\ fisso
Overriding model.yaml nc=80 with nc=2


optimizer: AdamW(lr=0.001667, momentum=0.9)
 with parameter groups 57 weight(decay=0.0),
  64 weight(decay=0.0005), 
  63 bias(decay=0.0)
Image sizes 640 train, 640 val
'''
