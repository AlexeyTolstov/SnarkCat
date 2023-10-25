import cv2

from ultralytics import YOLO
import ultralytics
ultralytics.checks()

model = YOLO('best.pt')

img_path = "test_img.jpg"
results = model.predict(source=img_path, conf=0.5)

print(results)

img = results[0].orig_img
classes = results[0].names
boxes_cls = results[0].boxes.cls
boxes_xy = results[0].boxes.xyxy

for i in range(len(boxes_cls)):
    cv2.rectangle(
        img,
        list(map(int, boxes_xy[i][:2].tolist())),
        list(map(int, boxes_xy[i][2:].tolist())),
        (0, 255, 0),
        3
    )

    xn, yn = map(int, boxes_xy[i][:2].tolist())
    cv2.putText(img, 
            classes[int(boxes_cls[i])],
            (xn, yn),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA)

cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()