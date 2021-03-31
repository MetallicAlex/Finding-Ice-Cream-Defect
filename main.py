from icecream_classification import IceCreamClassifier


if __name__ == '__main__':
    ice_cream_classifier = IceCreamClassifier()
    ice_cream_classifier.connect_to_thermal_camera('rtsp://Univer:Univer2021@10.162.0.233:554/Streaming/Channels/201')
    ice_cream_classifier.save_data('datasets/03 dataset/train')