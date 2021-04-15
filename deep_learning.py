import tensorflow as tf
from tensorflow.keras.layers import Dense
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def train_test_split(filename: str, test_size: float = 0.2):
    dataset = pd.read_csv(filename, sep=';')
    train_set = dataset.sample(frac=1 - test_size, random_state=200)
    test_set = dataset.drop(train_set.index)
    return train_set['image'].values, np.array(train_set['class'].values), test_set['image'].values, np.array(test_set['class'].values)


def preprocessing(filename: str):
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
    # ICE-CREAM CLASSIFICATION
    return number_blue_pixels / number_red_pixels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = train_test_split('datasets/04 dataset/dataset4.csv')
    train_ratios = []
    test_ratios = []
    for filename in train_images:
        train_ratios.append(preprocessing(f'datasets/{filename}'))
    for filename in test_images:
        test_ratios.append(preprocessing(f'datasets/{filename}'))
    train_ratios = np.array(train_ratios)
    test_ratios = np.array(test_ratios)

    model = tf.keras.Sequential([
        tf.keras.Input(1),
        Dense(20, activation='relu'),
        Dense(10, activation='relu'),
        Dense(4)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_ratios, train_labels, epochs=5000)
    model.evaluate(test_ratios, test_labels)

