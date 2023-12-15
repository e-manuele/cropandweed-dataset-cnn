import shutil
from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__':
    best_model_path = "runs/detect/train/weights/best.pt"
    image_path = 'ave-0037-0001.jpg'
    target_path = 'data/images/' + image_path
    model = YOLO(best_model_path)
    results = model(target_path)  # , conf=.1)
    shutil.copy(target_path, "target.jpg")

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('result.jpg')
    print("Da confrontare con visualize_annotations.py")
