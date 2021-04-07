import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


class IceCreamClassifier:
    def __init__(self):
        self._thermal_camera = cv2.VideoCapture
        self._optical_camera = cv2.VideoCapture

    def connect_to_optical_camera(self, camera: str):
        self._optical_camera = cv2.VideoCapture(camera)

    def connect_to_thermal_camera(self, camera: str):
        self._thermal_camera = cv2.VideoCapture(camera)

    def disconnect_to_optical_camera(self):
        pass

    def disconnect_to_thermal_camera(self):
        self._thermal_camera.release()
        cv2.destroyAllWindows()

    def save_data(self, filepath: str):
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        images_list = [int(os.path.basename(image.replace('.jpg', ''))) for image in glob.glob(f'{filepath}/*.jpg')]
        print(images_list)
        if len(images_list) > 0:
            number_image = max(images_list) + 1
        else:
            number_image = 0
        while True:
            ret, frame = self._thermal_camera.read()
            if ret:
                cv2.imshow('Thermal Camera', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'{filepath}/{number_image}.jpg', frame)
                    print(number_image)
                    number_image += 1
            else:
                self.disconnect_to_thermal_camera()
                break

    def show_histogram(self, filename: str):
        image = cv2.imread(filename)
        figure, axes = plt.subplots(1, 2, constrained_layout=True)
        # plt.hist(image.ravel(), 256,)
        plt.imshow(image)
        plt.show()

    def read_dataset(self, filename: str):
        pass

    def algorithm1(self):
        pass
