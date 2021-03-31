import cv2
import numpy as np
import glob
from tqdm import tqdm
import imutils
import time

speed = 100


def create_video(dataset: str, video: str, size: tuple):
    out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
    for filename in tqdm(glob.glob(f'{dataset}/*.jpg')):
        frame = cv2.imread(filename)
        out.write(frame)
    out.release()


def classify(number_normal_pixels, number_defect_pixels):
    ratio = number_normal_pixels / number_defect_pixels
    if 10 <= ratio <= 65:
        return 'Normal Ice-Cream'
    elif 1 <= ratio < 10:
        return '0.5 Ice-Cream'
    elif ratio < 1:
        return 'Empty Ice-Cream'
    elif ratio > 65:
        return '1.5 Ice-Cream'


def set_speed(value):
    global speed
    speed = max(value, 1)


def find_ice_cream(filename):
    cap = cv2.VideoCapture(filename)
    cv2.namedWindow(filename)
    avr_time = []
    cv2.createTrackbar('Speed', filename, speed, 100, set_speed)
    flag = False
    number_ice_cream = 0
    number_frame = 0
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # frame = cv2.imread(filename)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = frame[60:180]
            cv2.line(roi, (160, 0), (160, 180), (0, 0, 255), 2)
            lower_range_normal = (0, 0, 0)
            upper_range_normal = (50, 50, 50)
            normal_region = cv2.inRange(roi, lower_range_normal, upper_range_normal)
            lower_range_defect = (180, 180, 180)
            upper_range_defect = (255, 255, 255)
            defect_region = cv2.inRange(roi, lower_range_defect, upper_range_defect)
            contours, _ = cv2.findContours(normal_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(defect_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours += contours2
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                roi2 = roi[y:y + h, x:x + w]
                normal_region2 = normal_region[y:y + h, x:x + w]
                defect_region2 = defect_region[y:y + h, x:x + w]
            # normal_region2 = cv2.morphologyEx(normal_region2, cv2.MORPH_CLOSE, kernel)
            # defect_region2 = cv2.morphologyEx(defect_region2, cv2.MORPH_CLOSE, kernel)
                number_defect_pixels = cv2.countNonZero(defect_region2)
                number_normal_pixels = cv2.countNonZero(normal_region2)
                max_pixels = roi2.shape[0] * roi2.shape[1]
            if number_defect_pixels/max_pixels > 0.005 and number_normal_pixels/max_pixels > 0.005:
                if h + w > 160 and not flag:
                    text = classify(number_normal_pixels, number_defect_pixels)
                    with open('result.txt', 'a+') as file:
                        file.write(f'{text}\n')
                    cv2.imwrite(f'result/{number_frame}_{text}.jpg', frame)
                    # cv2.rectangle(roi, (x, y), (x + w, y + h), 255, 2)
                    cv2.putText(frame, text, (0, 15),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(0, 255, 0), thickness=1)
                    cv2.putText(frame, f'{number_normal_pixels / max_pixels * 100}', (0, 30),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(0, 255, 0), thickness=1)
                    cv2.putText(frame, f'{number_defect_pixels / max_pixels * 100}', (0, 45),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
                                color=(0, 255, 0), thickness=1)
                    flag = True
                    number_ice_cream += 1
            else:
                flag = False
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            roi = np.vstack((roi, normal_region, defect_region))
            if contours:
                ice_cream = np.vstack((roi2, normal_region2, defect_region2))
            cv2.imshow(filename, frame)
            cv2.imshow('ROI', roi)
            if contours:
                cv2.imshow('ICE-CREAM', ice_cream)
            avr_time.append(time.time() - start)
            number_frame += 1
            if cv2.waitKey(int(speed)) & 0xFF == ord('q'):
                print(number_ice_cream)
                avr_time = np.array(avr_time)
                print(np.average(avr_time))
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # create_video('showVideo1/images_1', 'dataset/001.avi', (320, 240))
    find_ice_cream('dataset/001.avi')
