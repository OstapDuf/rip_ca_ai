import cv2
import math
import cvzone
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

model = YOLO('yolov8n.pt')
classNames = model.names

def process_frame(img):
    results = model(img)
    count_cars = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if 0 <= cls < len(classNames):
                if classNames[cls] == "car":
                    count_cars += 1
                elif classNames[cls] == "tram":
                    count_cars += 5
                elif classNames[cls] == "bus":
                    count_cars += 3

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=15)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            else:
                print(f"Warning: Detected class index {cls} is out of range for classNames.")

    return img, count_cars

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skip = 5
    frame_id = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            success, img = cap.read()
            if not success:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            img = cv2.resize(img, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            future = executor.submit(process_frame, img)
            img, count_cars = future.result()

            text = f'Cars: {count_cars}'
            text_position = (50, 50)
            text_color = (0, 255, 0)
            cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    img = cv2.imread(image_path)
    img, count_cars = process_frame(img)

    text = f'Cars: {count_cars}'
    text_position = (50, 50)
    text_color = (0, 255, 0)
    cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_type = input("Введіть 'video' для обробки відео, 'image' для обробки зображення або 'stream' для трансляції з камери телефону: ").strip().lower()
if input_type == 'video':
    video_path = '7.mp4'
    process_video(video_path)
elif input_type == 'image':
    image_path = '2.jpg'
    process_image(image_path)
elif input_type == 'stream':
    rtsp_url = 'rtsp://admin:1234@10.137.74.188:8554/live'
    process_video(rtsp_url)
else:
    print("Невірний ввід. Введіть 'video', 'image' або 'stream'.")
