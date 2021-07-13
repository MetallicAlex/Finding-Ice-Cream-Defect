import numpy as np
import cv2
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
from pathlib import Path
import datetime

import models


class System:
    def __init__(self):
        self.modbus = ModbusClient
        self.gray_camera = None
        self.color_camera = None
        self.gray_image = None
        self.sample_image = cv2.imread('datasets/sample.png')
        self.x = 355
        self.y = 940
        self.height = 550
        self.width = 300
        self.sample_image = self.sample_image[self.y:self.y + self.height, self.x: self.x + self.width]
        self.color_image = None
        self.ice_cream = None

    def set_modbus_client(self, host: str = '127.0.0.1', port: int = 502):
        self.modbus = ModbusClient(host=host, port=port)

    def connect_camera(self, url: str, url2: str = None):
        self.gray_camera = cv2.VideoCapture(url)
        self.color_camera = cv2.VideoCapture(url2)

    def identify_ice_cream_defect(self):
        start_time = datetime.datetime.now()
        ret1, self.gray_image = self.gray_camera.retrieve(self.gray_camera.grab())
        ret2, self.color_image = self.color_camera.retrieve(self.color_camera.grab())
        if ret1 and ret2:
            with models.get_session() as session:
                if self.is_point_offset():
                    try:
                        self.ice_cream = models.IceCream()
                        self.ice_cream.created_at = start_time
                        self.ice_cream.cost = self.cost()
                        self.ice_cream.defect = self.is_ice_cream_defect(self.ice_cream.cost)
                        self.ice_cream.image = self.save_images(start_time)
                        self.send_signal_to_controller(0, self.ice_cream.defect, 1)
                    except Exception as e:
                        self.ice_cream.payload = e
                    finally:
                        self.ice_cream.runtime = datetime.datetime.now() - start_time
                        session.add(self.ice_cream)

    def is_ice_cream_defect(self, cost: float):
        if 0.85 <= cost <= 1.6:
            return True
        else:
            return False

    def is_point_offset(self):
        image = self.color_image[self.y:self.y + self.height, self.x: self.x + self.width]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_range = np.array([0, 50, 50])
        upper_range = np.array([10, 255, 255])
        mask_red = cv2.inRange(image, lower_range, upper_range)
        lower_range = np.array([170, 50, 50])
        upper_range = np.array([180, 255, 255])
        mask_red2 = cv2.inRange(image, lower_range, upper_range)
        mask_red = cv2.bitwise_or(mask_red, mask_red2)
        number_red_pixels = cv2.countNonZero(mask_red)
        if number_red_pixels > 0:
            return True
        return False

    def cost(self):
        roi = self.gray_image
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
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask = cv2.bitwise_and(roi, mask)
        # ICE-CREAM CLASSIFICATION
        if number_red_pixels == 0:
            number_red_pixels = 1
        return number_blue_pixels / number_red_pixels

    def save_images(self, start_time: datetime.datetime):
        start_time = start_time.strftime('%Y-%m-%d %H-%M-%S-%f')
        start_time = start_time.split(' ')
        folder = start_time[0]
        filename = start_time[1]
        Path('images').mkdir(parents=True, exist_ok=True)
        Path(f'images/{folder}').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'images/{folder}/{filename}_gray.jpg', self.gray_image)
        cv2.imwrite(f'images/{folder}/{filename}_color.jpg', self.color_image)
        return f'images/{folder}/{filename}'

    def send_signal_to_controller(self, address: int, value: int, unit: int):
        self.modbus.write_coil(address, value, unit=unit)