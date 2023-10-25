from os import listdir, path, rename
import cv2
import albumentations as A

input_folder = "D:/Yolov8 Model/Dataset/Images/"

transform = A.Compose([
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1)
])
n = 0
for image_name in listdir(input_folder):
    if image_name.endswith((".jpg", ".png")):
        image_path = path.join(input_folder, image_name)
        rename(image_path, f"{n}.jpg")
        n += 1

print("Преобразование завершено.")