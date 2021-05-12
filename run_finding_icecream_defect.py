# python run_finding_icecream_defect.py -r roi.json -a parameters_algorithm.json -c '127.0.0.1' -u 'rtsp://Univer:Univer2021@194.158.204.222:554/Streaming/Channels/201'

import cv2
import numpy as np
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import datetime
import time
from pathlib import Path
import json
import argparse


def log(filename: str, text: str):
    with open(filename, 'w+') as file:
        file.write(text)


def finding_ice_cream(capture: cv2.VideoCapture, roi_lines: dict, parameters: dict,
                      modbus_client: ModbusClient, address: int, unit: int):
    start = time.time()
    ret, frame = capture.retrieve(capture.grab())
    if ret:
        roi = frame[roi_lines['lines1']['y']:roi_lines['lines2']['y']]
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
        ratio = number_blue_pixels / number_red_pixels
        defect = is_ice_cream_defect(ratio, parameters)
        save_frame(frame, 'frame')
        save_frame(mask, 'mask')
        log('events.log', f'[{datetime.datetime.now()}][{time.time() - start}] ice-cream defect is {defect}\n')
        if defect:
            send_result_to_controller(modbus_client, address, 1, unit)
        else:
            send_result_to_controller(modbus_client, address, 0, unit)
    else:
        log('errors.log', f'[{datetime.datetime.now()}] camera is not connected')


def is_ice_cream_defect(ratio: float, parameters: dict):
    if parameters['normal']['min'] <= ratio < parameters['normal']['max']:
        ice_cream_class = False
    else:
        ice_cream_class = True
    return ice_cream_class


def save_frame(frame: np.array, suffix: str):
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S-%f')
    dt = dt.split(' ')
    folder = dt[0]
    filename = dt[1] + f'_{suffix}.jpg'
    Path('images').mkdir(parents=True, exist_ok=True)
    Path(f'images/{folder}').mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'images/{folder}/{filename}', frame)


def send_result_to_controller(modbus_client: ModbusClient, address: int, value: int, unit: int):
    modbus_client.write_coil(address, value, unit=unit)
    log('events.log', f'[{datetime.datetime.now()}] write coil to unit={unit}, address={address} value: {value}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--roi", required=True,
                    help="path to roi.json")
    ap.add_argument("-a", "--algorithm", required=True,
                    help="path to algorithm.json")
    ap.add_argument("-u", "--url", required=True,
                    help="path to ip-camera")
    ap.add_argument("-c", "--client", required=True,
                    help="modbus client ip-address")
    args = vars(ap.parse_args())
    with open(args['roi']) as file:
        lines = json.load(file, strict=False)
    with open(args['algorithm']) as file:
        algorithm = json.load(file, strict=False)
    cap = cv2.VideoCapture(args['url'])
    client = ModbusClient(args['client'], port=5020)
    client.connect()
    while True:
        finding_ice_cream(
            capture=cap,
            roi_lines=lines,
            parameters=algorithm,
            modbus_client=client,
            address=0,
            unit=1
        )
