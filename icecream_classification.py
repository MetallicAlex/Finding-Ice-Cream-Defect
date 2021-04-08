import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class IceCreamClassifier:
    def __init__(self):
        self._thermal_camera = cv2.VideoCapture
        self._optical_camera = cv2.VideoCapture
        self._dataset = pd.DataFrame
        self._correctly_classify_ice_cream = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0
        }
        self._amount_ice_cream_by_class = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0
        }
        self._correctly_normal_defect_ice_cream = {
            'defect': 0,
            'normal': 0
        }
        self._amount_normal_defect_ice_cream = {
            'defect': 0,
            'normal': 0
        }

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
        self._dataset = pd.read_csv(filename, sep=';')
        unique, counts = np.unique(self._dataset['class'].values, return_counts=True)
        self._amount_ice_cream_by_class = dict(zip(unique, counts))
        for key, value in self._amount_ice_cream_by_class.items():
            if int(key) == 0:
                self._amount_normal_defect_ice_cream['normal'] = value
            else:
                self._amount_normal_defect_ice_cream['defect'] += value
        print(self._dataset)
        print(self._amount_ice_cream_by_class)
        print(self._amount_normal_defect_ice_cream)

    def test(self):
        ice_cream_classes = []
        ice_creams_correctly = []
        ice_cream_ratios = []
        for index, row in self._dataset.iterrows():
            ice_cream_class, ratio = self.algorithm1(f"datasets/{row['image']}")
            ice_cream_classes.append(ice_cream_class)
            ice_cream_ratios.append(ratio)
            if ice_cream_class == row['class']:
                ice_creams_correctly.append(True)
                self._correctly_classify_ice_cream[str(row['class'])] += 1
            else:
                ice_creams_correctly.append(False)
            if ice_cream_class == row['class'] and ice_cream_class == 0:
                self._correctly_normal_defect_ice_cream['normal'] += 1
            elif row['class'] > 0 and ice_cream_class > 0:
                self._correctly_normal_defect_ice_cream['defect'] += 1
        self._dataset['classify'] = ice_cream_classes
        self._dataset['correctly'] = ice_creams_correctly
        self._dataset['ratio (Blue/Red)'] = ice_cream_ratios
        print(self._correctly_classify_ice_cream)
        print(self._correctly_normal_defect_ice_cream)
        self._dataset.to_csv('result.csv', sep=';')

    def plot_accuracy(self):
        accuracy = [
            (self._correctly_classify_ice_cream[str(key)] / value) * 100
            for key, value in self._amount_ice_cream_by_class.items()
        ]
        general_accuracy = 0
        for value in self._correctly_classify_ice_cream.values():
            general_accuracy += value
        general_accuracy = (general_accuracy / len(self._dataset.index)) * 100
        accuracy.append(general_accuracy)
        print(accuracy)
        y = ['Нормальное', 'Пустое', 'Половина', 'Полтора', 'Общая']
        figure, ax = plt.subplots(figsize=(16, 9))
        ax.barh(y, accuracy, height=0.5)
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     str(round((i.get_width()), 2)),
                     fontsize=16, fontweight='bold',
                     color='grey')
        ax.grid(True)
        plt.title('Точность обнаружения мороженного')
        plt.show()

    def plot_accuracy_defect_normal(self):
        accuracy = [
            self._correctly_normal_defect_ice_cream['defect']
            * 100 / self._amount_normal_defect_ice_cream['defect'],
            self._correctly_normal_defect_ice_cream['normal']
            * 100 / self._amount_normal_defect_ice_cream['normal'],
            (self._correctly_normal_defect_ice_cream['normal'] + self._correctly_normal_defect_ice_cream['defect'])
            * 100 / len(self._dataset.index)
        ]
        print(accuracy)
        y = ['Брак', 'Нормальное', 'Общая']
        figure, ax = plt.subplots(figsize=(16, 9))
        ax.barh(y, accuracy, height=0.5)
        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     str(round((i.get_width()), 2)),
                     fontsize=16, fontweight='bold',
                     color='grey')
        ax.grid(True)
        plt.title('Точность обнаружения мороженного (Брак/Нормальное)')
        plt.show()

    def algorithm1(self, filename: str):
        frame = cv2.imread(filename)
        roi = frame[80:200]
        image_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # RED COLOR
        lower_range = np.array([0, 50, 50])
        upper_range = np.array([10, 255, 255])
        mask_red = cv2.inRange(image_hsv, lower_range, upper_range)
        lower_range = np.array([170, 50, 50])
        upper_range = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(image_hsv, lower_range, upper_range)
        mask_red = cv2.bitwise_or(mask_red, mask_red2)
        number_red_pixels = cv2.countNonZero(mask_red)
        # BLUE COLOR
        lower_range = np.array([110, 50, 50])
        upper_range = np.array([130, 255, 255])
        mask_blue = cv2.inRange(image_hsv, lower_range, upper_range)
        number_blue_pixels = cv2.countNonZero(mask_blue)
        # DETECT ICE-CREAM
        mask = cv2.bitwise_or(mask_blue, mask_red)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.bitwise_and(roi, mask)
        # ICE-CREAM CLASSIFICATION
        ratio = number_blue_pixels / number_red_pixels
        if ratio >= 1.6:
            ice_cream_class = 3
        elif 0.85 <= ratio < 1.6:
            ice_cream_class = 0
        elif 0.3 <= ratio < 0.85:
            ice_cream_class = 2
        elif ratio < 0.3:
            ice_cream_class = 1
        return ice_cream_class, ratio
        # cv2.putText(mask, f'Max area: {cv2.contourArea(max_contour)}', (0, 15),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
        #             color=(0, 255, 0), thickness=1)
        # cv2.putText(mask, f'Red pixels:  {number_red_pixels}', (0, 30),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
        #             color=(0, 255, 0), thickness=1)
        # cv2.putText(mask, f'Blue pixels: {number_blue_pixels}', (0, 45),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1,
        #             color=(0, 255, 0), thickness=1)
        # cv2.drawContours(mask, contours, -1, (0, 255, 0), 1)
        # output = np.zeros((120, 640, 3), dtype='uint8')
        # output[0:120, 0:320] = roi
        # output[0:120, 320:640] = mask
        # cv2.imshow(filename, output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return '1'
