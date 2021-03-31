import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class IceCreamClassifier:
    def __init__(self):
        self._thermal_camera = cv2.VideoCapture
        self._optical_camera = cv2.VideoCapture

    def connect_to_thermal_camera(self, camera: str):
        self._thermal_camera = cv2.VideoCapture(camera)

    def save_data(self, filepath: str):
        number_frame = 0
        if not os.path.isdir(filepath):
            print(False)
            os.mkdir(filepath)
        else:
            print(True)
        while True:
            ret, frame = self._thermal_camera.read()
            if ret:
                cv2.imshow('Thermal Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'{filepath}/{number_frame}')
                    number_frame += 1
            else:
                self._thermal_camera.release()
                break

    def connect_to_optical_camera(self, camera: str):
        self._optical_camera = cv2.VideoCapture(camera)

    def read_dataset(self):
        pass