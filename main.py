
from system import System

if __name__ == '__main__':
    system = System()
    system.set_modbus_client()
    system.modbus.connect()
    system.connect_camera(url='rtsp://univer:Univer123@192.168.48.248:554/Streaming/Channels/201',
                          url2='rtsp://univer:Univer123@192.168.48.248:554/Streaming/Channels/101')
    while True:
        system.identify_ice_cream_defect()