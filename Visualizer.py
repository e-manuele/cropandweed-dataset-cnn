import shutil
from PIL import Image
from ultralytics import YOLO

best_model_path = "runs/detect/train/weights/best.pt"
image_path = 'ave-0035-0005.jpg'
target_path = 'data/images/'+image_path
model = YOLO(best_model_path)
results = model(target_path)


shutil.copy(target_path, "target.jpg")
# Show the results
for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('result.jpg')


