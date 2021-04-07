import cv2
import datetime

if __name__ == '__main__':
    cap = cv2.VideoCapture('rtsp://Univer:Univer2021@192.168.43.228:554/Streaming/Channels/201')
    while True:
        ret, frame = cap.retrieve(cap.grab())
        if ret:
            now = datetime.datetime.now()
            cv2.imwrite(f'datasets\\04 dataset\\train\\{now.strftime("%H-%M-%S.%f")}.jpg', frame)
            cv2.imshow('thermal', frame)
        else:
            cap.release()
            cap = cv2.VideoCapture("rtsp://Univer:Univer2021@192.168.43.228:554/Streaming/Channels/201")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
