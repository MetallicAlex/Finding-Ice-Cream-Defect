# python run_finding_icecream_defect.py -r roi.json -a parameters_algorithm.json -c 127.0.0.1 -u rtsp://Univer:Univer2021@194.158.204.222:554/Streaming/Channels/201

import cv2
import numpy as np
from pymodbus.client.sync import ModbusTcpClient as ModbusClient
import datetime
import time
from pathlib import Path
import json
import argparse


def log(filename: str, text: str):
    with open(filename, 'a+') as file:
        file.write(text)


def finding_ice_cream(frame: np.array, roi_lines: dict, parameters: dict,
                      modbus_client: ModbusClient, address: int, unit: int):
    start = time.time()
    roi = frame[roi_lines['line1']['y']:roi_lines['line2']['y']]
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
    ratio = number_blue_pixels / number_red_pixels
    defect = is_ice_cream_defect(ratio, parameters)
    save_frame(frame, 'frame')
    save_frame(mask, 'mask')
    log('events.log', f'[{datetime.datetime.now()}][{time.time() - start}] ice-cream defect is {defect}\n')
    if defect:
        send_result_to_controller(modbus_client, address, 1, unit)
    else:
        send_result_to_controller(modbus_client, address, 0, unit)


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
    print(args['url'])
    cap = cv2.VideoCapture(args['url'])
    # cap = cv2.VideoCapture('rtsp://Univer:Univer2021@194.158.204.222:554/Streaming/Channels/201')
    client = ModbusClient(args['client'], port=502)
    client.connect()
    times = []
    for _ in range(100):
        time.sleep(0.2)
        start = time.time()
        ret, frame = cap.read()
        print(ret)
        if ret:
            cv2.imshow('Frame', frame)
            finding_ice_cream(
                frame=frame,
                roi_lines=lines,
                parameters=algorithm,
                modbus_client=client,
                address=0,
                unit=1
            )
        else:
            log('errors.log', f'[{datetime.datetime.now()}] camera is not connected')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        times.append((time.time() - start)*1000)
        log('time.log', f'{time.time() - start}\n')
    print(np.average(times))
    cap.release()
    cv2.destroyAllWindows()
