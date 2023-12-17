import shutil
from PIL import Image
from ultralytics import YOLO
import subprocess

if __name__ == '__main__':
    best_model_path = "runs/detect/train/weights/best.pt"
    file_name = 'ave-0058-0013'
    image_path = file_name + '.jpg'
    target_path = 'data/images/' + image_path
    model = YOLO(best_model_path)

    inference = model(target_path, visualize=True)#, save=True, save_crop=True
    '''

    results = model(target_path)  # , conf=.1)
    shutil.copy(target_path, "target.jpg")
    '''
    for data in inference:
        im_array = data.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('output_evaluation/result.jpg')
        print("img saved!")
    #--dataset CropOrWeed2 --filter ave-0058-0013
    #subprocess.call(["python", "cnw/visualize_annotations.py", "--dataset", "CropOrWeed2", "--filter", file_name])
    print("Da confrontare risultati presenti in cartella output-evaluation")


# https://www.youtube.com/watch?v=dVE0THUbPH4
#https://www.youtube.com/watch?v=Z-65nqxUdl4
