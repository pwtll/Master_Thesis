"""
This code originates from Wang et al. [1] Github-repository: PhysRecorder (https://github.com/KegangWangCCNU/PhysRecorder)
[1] Kegang Wang, Yantao Wei, Mingwen Tong, Jie Gao, Yi Tian, YuJian Ma, & ZhongJin Zhao. (2023). PhysBench: A Benchmark Framework for Remote Physiological Sensing with New Dataset and Baseline.

Changes were made to include logging of subject information, like:
    - (Fitzpatrick) skin_tone, gender, age, (wears) glasses, (wears) hair_cover, (wears) makeup, (active) sunlight_illumination

Additional Changes include the parallel recording of two cameras, which are connected via USB:
    1. Logitech C930e webcam with a resolution of 640x480 pixel @ 30 fps
    2. Sigma fp camera with a resolution of 1920x1080 pixel @ 29.97 fps

Additional change includes the support of Pulox PO-250 fingerclip pulse oximeter as serial device.
The code base for the implementation of Pulox-support originates from: https://github.com/timegrid/CMS50Dplus7

Some parts of the bounding box' movement coordination were written together with ChatGPT.
"""

import sys
import time
import random
import csv
import argparse
import threading
import tkinter
from tkinter import messagebox, simpledialog, filedialog

import serial
from dateutil import parser as dateparser

import datetime
import csv

import hid, time, threading, cv2, os
from PyCameraList.camera_device import list_video_devices
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor()


arduino_port = 'COM4'
arduino_baudrate = 9600
arduino_update_sent = False
last_update_time = None

arduino = None

try:
    arduino = serial.Serial(arduino_port, arduino_baudrate)
except Exception as e:
    print(e)

serial_port = 'COM3'
cam_id = 1
swap_camera_order = False
cam_idx = [0, 1]
use_sigma = False
frame_res_camera_0_logitech = [640, 480]
frame_res_camera_1_sigma = [1920, 1080]

res = (640, 480)
fourcc = 'YUY2'
fps = 30
save_fourcc = []
frames = []
frames_sigma = []
start_time, end_time = 0, 0
alive = True
dir_path = ''
recordings_dir = "recordings"

# load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# variables to store bounding box coordinates
dynamic_x, dynamic_y, dynamic_w, dynamic_h = None, None, None, None
use_dynamic_box = False  # Flag to determine when to use the dynamic box
frame_counter = 0
wait_counter = 0
movement_increment = 0.07/4  # 7% movement
move_left = True
move_back = False
move_top = False
move_bottom = False
move_right = False
move_center = False
start_time_bb = None
start_time_rot = None
movement_duration = 3  # Movement duration in seconds


class DataPoint():
    datatype = ''
    specs = {}  # {package_type: package_length, ...}
    attributes = []  # [(attribute, string, csvheader), ...]

    def __init__(self, package_type, package, time=False):
        self.time = time and time or datetime.datetime.now()

        # package type
        if package_type not in self.specs:
            raise ValueError("Invalid package type.")
        self.package_type = package_type

        # packet length
        if len(package) != self.specs[self.package_type]:
            raise ValueError("Invalid package length.")

        # set data
        self.set_package(package_type, package, time)

    def __repr__(self):
        hexBytes = ['0x{0:02X}'.format(byte) for byte in self.get_package()]
        return "{}({}, [{}], {})".format(
            self.__class__.__name__, self.package_type, ', '.join(hexBytes),
            repr(self.time))

    def __str__(self):
        return ",\n".join([attr[1] for attr in self.attributes]).format(
            **[getattr(self, attr[0]) for attr in self.attributes]
        )

    @classmethod
    def get_attribute_names(cls):
        return [attr[0] for attr in cls.attributes]

    @classmethod
    def get_csv_header(cls):
        return [attr[2] for attr in cls.attributes]

    def get_csv_data(self):
        return [getattr(self, attr[0]) for attr in self.attributes]

    def set_csv_data(self, data):
        for attr, _, key in self.attributes:
            if key in data:
                value = data[key]
                if isinstance(value, float):
                    value = int(value)
                if attr == 'time':
                    value = dateparser.parse(value)
                setattr(self, attr, value)

    def get_dict_data(self):
        ret = dict()
        for n, d in zip(self.get_csv_header(), self.get_csv_data()):
            ret[n] = d
        return ret

    def set_package(package_type, package, time):
        raise NotImplementedError('set_package() not implemented.')

    def get_package(self):
        raise NotImplementedError('get_package() not implemented.')


class RealtimeDataPoint(DataPoint):
    datatype = 'realtime'
    specs = {  # {package_type: package_length, ...}
        0x01: 7
    }
    attributes = [  # [(attribute, string, csvheader), ...]
        ('time',               "Time = {}",               "Time"),
        ('spO2',               "SpO2 = {}%",              "SpO2"),
        ('pulse_rate',         "Pulse Rate = {} bpm",     "PulseRate"),
        ('pulse_waveform',     "Pulse Waveform = {}",     "PulseWaveform"),
        ('pulse_beep',         "Pulse Beep = {}",         "PulseBeep"),
        ('bar_graph',          "Bar Graph = {}",          "BarGraph"),
        ('pi',                 "PI = {}%",                "Pi"),
        ('signal_strength',    "Signal Strength = {}",    "SignalStrength"),
        ('probe_error',        "Probe Error = {}",        "ProbeError"),
        ('low_spO2',           "Low SpO2 = {}",           "LowSpO2"),
        ('searching_too_long', "Searching Too Long = {}", "SearchingTooLong"),
        ('searching_pulse',    "Searching Pulse = {}",    "SearchingPulse"),
        ('spO2_invalid',       "SpO2 Invalid = {}",       "SpO2Invalid"),
        ('pulse_rate_invalid', "Pulse Rate Invalid = {}", "PulseRateInvalid"),
        ('pi_valid',           "PI Valid = {}",           "PiValid"),
        ('pi_invalid',         "PI Invalid = {}",         "PiInvalid"),
        ('reserved',           "Reserved = {}",           "Reserved"),
        ('datatype',           "Data Type = {}",          "DataType"),
        ('package_type',       "Package Type = {}",       "PackageType"),
    ]

    def set_package(self, package_type, package, time):
        # packet byte 2 / package byte 0
        self.signal_strength = package[0] & 0x0f
        self.searching_too_long = (package[0] & 0x10) >> 4
        self.low_spO2 = (package[0] & 0x20) >> 5
        self.pulse_beep = (package[0] & 0x40) >> 6
        self.probe_error = (package[0] & 0x80) >> 7

        # packet byte 3 / package byte 1
        self.pulse_waveform = package[1] & 0x7f
        self.searching_pulse = (package[1] & 0x80) >> 7

        # packet byte 4 / package byte 2
        self.bar_graph = package[2] & 0x0f
        self.pi_valid = (package[2] & 0x10) >> 4
        self.reserved = (package[2] & 0xe0) >> 5

        # packet byte 5 / package byte 3
        self.pulse_rate = package[3]
        self.pulse_rate_invalid = int(self.pulse_rate == 0xff)

        # packet byte 6 / package byte 4
        self.spO2 = package[4]
        self.spO2_invalid = int(self.spO2 == 0x7f)

        # packet byte 7-8 / package byte 5-6
        self.pi = package[6] << 8 | package[5]
        self.pi_invalid = int(self.pi == 0xffff)

    def get_package(self):
        package = [0] * self.specs[self.package_type]

        # packet byte 2 / package byte 0
        package[0] = self.signal_strength & 0x0f
        if self.searching_too_long:
            package[0] |= 0x10
        if self.low_spO2:
            package[0] |= 0x20
        if self.pulse_beep:
            package[0] |= 0x40
        if self.probe_error:
            package[0] |= 0x80

        # packet byte 3 / package byte 1
        package[1] = self.pulse_waveform & 0x7f
        if self.searching_pulse:
            package[1] |= 0x80

        # packet byte 4 / package byte 2
        package[2] = self.bar_graph & 0x0f
        if self.pi_valid:
            package[2] |= 0x10
        package[2] |= (self.reserved << 5) & 0xe0

        # packet byte 5 / package byte 3
        package[3] = self.pulse_rate & 0xff

        # packet byte 6 / package byte 4
        package[4] = self.spO2 & 0xff

        # packet byte 7-8 / package byte 5-6
        package[5] = self.pi & 0x00ff
        package[6] = (self.pi & 0xff00) >> 8

        return package


class StorageDataPoint(DataPoint):
    datatype = 'storage'
    specs = {  # {package_type: package_length, ...}
        0x0f: 2,  # one package of 6 bytes split into 3 datapoints
        0x09: 4,
    }
    attributes = [  # [(attribute, string, csvheader), ...]
        ('time', "Time = {}", "Time"),
        ('spO2', "SpO2 = {}%", "SpO2"),
        ('pulse_rate', "Pulse Rate = {} bpm", "PulseRate"),
        ('pi', "PI = {}%", "Pi"),
        ('pi_support', "PI Support = {}", "PiSupport"),
        ('pulse_rate_invalid', "Pulse Rate Invalid = {}", "PulseRateInvalid"),
        ('spO2_invalid', "SpO2 Invalid = {}", "SpO2Invalid"),
        ('pi_invalid', "PI Invalid = {}", "PiInvalid"),
        ('datatype', "Data Type = {}", "DataType"),
        ('package_type', "Package Type = {}", "PackageType"),
    ]

    def set_package(self, package_type, package, time):
        # pi support
        self.pi_support = 0
        if self.package_type == 0x09:
            self.pi_support = 1

        # packet byte 2|4|6 / package byte 0
        self.spO2 = package[0] & 0xff
        self.spO2_invalid = int(self.spO2 == 0x7f)

        # packet byte 3|5|7 / package byte 1
        self.pulse_rate = package[1] & 0xff
        self.pulse_rate_invalid = int(self.pulse_rate == 0xff)

        # packet byte 4-5 / package byte 2-3
        if self.pi_support:
            self.pi = package[3] << 8 | package[2]
            self.pi_invalid = int(self.pi == 0xffff)
        else:
            self.pi = "-"
            self.pi_invalid = "-"

    def get_package(self):
        package = [0] * self.specs[self.package_type]

        # packet byte 2|4|6 / package byte 0
        package[0] = self.spO2 & 0xff

        # packet byte 3|5|7 / package byte 1
        package[1] = self.pulse_rate & 0xff

        # packet byte 4-5 / package byte 2-3
        if self.pi_support:
            package[2] = self.pi & 0x00ff
            package[3] = (self.pi & 0xff00) >> 8

        return package


class CMS50Dplus():
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.5,
                 connect=True):  # (self, port='/dev/ttyUSB0', baudrate=115200, timeout=0.5, connect=True):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        self.storage_time_interval = datetime.timedelta(seconds=1)
        self.connection = None
        if connect:
            self.connect()

    def __del__(self):
        self.disconnect()

    @staticmethod
    def set_bit(byte, value=1, index=7):
        mask = 1 << index
        byte &= ~mask
        if value:
            byte |= mask
        return byte

    @classmethod
    def decode_package(cls, packets):
        # check packet length
        if len(packets) < 3:
            raise ValueError("Package too short to decode.")
        if len(packets) > 9:
            raise ValueError("Package too long to decode")

        # check synchronization bits
        if packets[0] & 0x80:
            raise ValueError("Invalid synchronization bit.")
        for byte in packets[1:]:
            if not byte & 0x80:
                raise ValueError("Invalid synchronization bit.")

        # define packet parts
        package_type = packets[0]
        high_byte = packets[1]
        package = packets[2:]

        # decode high byte
        for idx, byte in enumerate(package):
            package[idx] = cls.set_bit(byte, high_byte & 0x01 << idx)

        return package_type, package

    @classmethod
    def encode_package(cls, package_type, package,
                       padding=0, padding_byte=0x00):
        # check package length
        if len(package) > 7:
            raise ValueError("Package too long to encode.")

        # define packet parts
        high_byte = 0x80
        package = package[:]

        # pad package
        if padding:
            if padding < len(package):
                raise ValueError("Padding too short.")
            if padding > 7:
                raise ValueError("Padding too long.")
            if padding > len(package):
                package += [padding_byte] * (padding - len(package))

        # encode high byte
        for idx, byte in enumerate(package):
            high_byte |= (byte & 0x80) >> (7 - idx)

        # set synchronization bits
        package_type = cls.set_bit(package_type, 0)
        for idx, byte in enumerate(package):
            package[idx] = cls.set_bit(byte)

        # compose packets
        packets = [package_type, high_byte] + package

        return packets

    def is_connected(self):
        if self.connection and self.connection.isOpen():
            return True
        return False

    def connect(self):
        if self.connection is None:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                xonxoff=1
            )
        elif not self.is_connected():
            self.connection.open()

    def disconnect(self):
        if self.is_connected():
            self.connection.close()

    def get_byte(self):
        char = self.connection.read()
        if len(char) == 0:
            return None
        else:
            return ord(char)

    def send_bytes(self, values):
        return self.connection.write(
            b''.join([
                chr(value & 0xff).encode('raw_unicode_escape')
                for value in values]))

    def expect_byte(self, value):
        while True:
            byte = self.get_byte()
            if byte is None:
                return False
            elif byte == value:
                return True

    def send_command(self, command, data=[]):
        package = self.encode_package(
            package_type=0x7d,  # command
            package=[command] + data, padding=7, padding_byte=0x00)
        self.send_bytes(package)
        self.connection.flush()

    def send_keepalive(self):
        now = datetime.datetime.now()
        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xaf)  # keepalive
            self.keepalive_timestamp = now

    def get_packets(self, amount=0):
        count = 0
        idx = 0
        packets = []
        while True:
            if not amount:
                self.send_keepalive()
            byte = self.get_byte()
            if byte is None:
                if len(packets[:idx]) < 3:
                    raise ValueError("Recieved too few bytes for packets.")
                if amount and count + 1 < amount:
                    raise ValueError("Recieved too few packets.")
                yield packets[:idx]
                break
            sync_bit = bool(byte & 0x80)
            if not sync_bit:
                if packets:
                    if len(packets[:idx]) < 3:
                        raise ValueError("Recieved too few bytes for packets.")
                    yield packets[:idx]
                    if amount:
                        count += 1
                        if count == amount:
                            break
                packets = [0x00] * 9
                idx = 0
            if idx > 8:
                raise ValueError("Received too many bytes for packets.")
            packets[idx] = byte
            idx += 1

    def get_packages(self, amount=0):
        for packets in self.get_packets(amount):
            package_type, package = self.decode_package(packets)
            if package_type == 0x0d:  # disconnect notice
                if package[0] in [0x00, 0x01]:
                    break
                raise ValueError(
                    "Received reasoncode 0x{:02X}".format(package[0]))
            yield package_type, package

    def get_realtime_data(self):
        try:
            self.connection.reset_input_buffer()
            self.send_command(0xa1)  # start realtime data
            for package_type, package in self.get_packages():
                yield RealtimeDataPoint(package_type, package)
        except KeyboardInterrupt:
            pass
        finally:
            self.send_command(0xa2)  # stop realtime data

    def get_storage_data(self, starttime=False,
                         user_index=0x01, storage_segment=0x01):
        if not starttime:
            starttime = datetime.datetime.now()
        try:
            self.connection.reset_input_buffer()
            self.send_command(  # start storage data
                0xa6, [user_index, storage_segment])
            for package_type, package in self.get_packages():
                if package_type == 0x0f:
                    if package[0] and package[1]:
                        yield StorageDataPoint(
                            package_type, package[0:2], time=starttime)
                        starttime += self.storage_time_interval
                    if package[2] and package[3]:
                        yield StorageDataPoint(
                            package_type, package[2:4], time=starttime)
                        starttime += self.storage_time_interval
                    if package[4] and package[5]:
                        yield StorageDataPoint(
                            package_type, package[4:6], time=starttime)
                        starttime += self.storage_time_interval
                else:
                    yield StorageDataPoint(
                        package_type, package, time=starttime)
                    starttime += self.storage_time_interval
        except KeyboardInterrupt:
            pass
        finally:
            self.send_command(0xa7)  # stop storage data

def print_realtime_data(port=serial_port):
    print("Saving live data...")
    print("Press CTRL-C / disconnect the device to terminate data collection.")

    oximeter = CMS50Dplus(port)
    datapoints = oximeter.get_realtime_data()
    try:
        for datapoint in datapoints:
            sys.stdout.write(
                "\rSignal: {:>2}"
                " | PulseRate: {:>3}"
                " | PulseWave: {:>3}"
                " | SpO2: {:>2}%"
                " | ProbeError: {:>1}".format(
                    datapoint.signal_strength,
                    datapoint.pulse_rate,
                    datapoint.pulse_waveform,
                    datapoint.spO2,
                    datapoint.probe_error))
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass


def get_realtime_data(port=serial_port):
    oximeter = CMS50Dplus(port)
    datapoints = oximeter.get_realtime_data()
    try:
        for datapoint in datapoints:
            sys.stdout.write(
                    "\rSignal: {:>2}"
                    " | PulseRate: {:>3}"
                    " | PulseWave: {:>3}"
                    " | SpO2: {:>2}%"
                    " | ProbeError: {:>1}".format(
                            datapoint.signal_strength,
                            datapoint.pulse_rate,
                            datapoint.pulse_waveform,
                            datapoint.spO2,
                            datapoint.probe_error))
            sys.stdout.flush()

            yield datapoint.pulse_rate, datapoint.pulse_waveform, datapoint.spO2
    except KeyboardInterrupt:
        pass

def connect_cam():
    global frames, cap, start_time_bb

    global dynamic_x, dynamic_y, dynamic_w, dynamic_h, use_dynamic_box, frame_counter, wait_counter, movement_increment, move_left, move_back, move_top, move_bottom, move_right, move_center, start_time_bb, movement_duration
    global start_time_rot
    rot_img = None

    global arduino_update_sent, arduino

    update_sent = False

    frames = []
    cap = None
    while alive:
        try:
            res_, fourcc_, fps_, cam_id_ = res, fourcc, fps, cam_id
            cam_id_ = cam_idx[swap_camera_order]
            # res_ = res = (1920, 1080)
            cap = cv2.VideoCapture(cam_id_, cv2.CAP_DSHOW)
            cap.set(5, fps)
            # cap.set(3, res[0])
            # cap.set(4, res[1])
            cap.set(3, frame_res_camera_0_logitech[0])
            cap.set(4, frame_res_camera_0_logitech[1])
            cap.set(6, cv2.VideoWriter.fourcc(*'YUY2'))
            w, h = cap.get(3), cap.get(4)
            # if h == 480:
            #     sel1.select()
            # elif h == 720:
            #     sel2.select()
            # elif h == 1080:
            #     sel3.select()
            fcc = cap.get(6)
            # if fcc == cv2.VideoWriter.fourcc(*'YUY2'):
            #     sel5.select()
            # elif fcc == cv2.VideoWriter.fourcc(*'MJPG'):
            #     sel4.select()
            r = ((640*480)/(w*h))**0.5
            # while (res_, fourcc_, fps_, cam_id_) == (res, fourcc, fps, cam_id) and alive:
            while (fourcc_, fps_) == (fourcc, fps) and alive:
                _, frame = cap.read()
                frame = cv2.flip(frame, 1)
                #print(cap.get(6)==cv2.VideoWriter.fourcc(*'YUY2'), time.time())
                #print(frame-cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420), cv2.COLOR_YUV2BGR_I420))
                frames.append((frame, time.time()))
                if recording and start_time:
                    while len(frames) > 100:
                        (f, t_) = frames.pop(0)
                        if dir_path and recording and time.time()<end_time:
                            with open(f'{dir_path}/missed_frames.csv', 'a') as csv:
                                csv.write(f'{t_}\n')
                else:
                    while len(frames) > 5:
                        frames.pop(0)
                f_ = cv2.resize(frame, (round(w*r), round(h*r)))
                if not start_time or not recording:
                    cv2.imshow('Logitech C930e', f_)
                    arduino.write(1)
                    # update_sent = False
                else:
                    if not update_sent:
                        arduino.write(0)
                        arduino_update_sent = True

                    t = time.time()-start_time
                    if int(t)%2:
                        cv2.putText(f_, f'{int(t)//60}:', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)
                    else:
                        cv2.putText(f_, f'{int(t)//60}', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)
                    cv2.putText(f_, f'{t%60:.1f}', (60, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)

                    # move the bounding box in scenario 8 in the following sceme:
                    # from origin to the left -> center -> top -> center -> bottom -> center -> right -> center -> repeat cycle
                    if text2.get() == 'v10' or text2.get() == 'v12':
                        '''
                        # Convert the frame to grayscale for face detection
                        gray = cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY)

                        # Detect faces in the frame
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        if len(faces) > 0: # if f_n == 30 and len(faces) > 0:
                            # save bounding box coordinates after 5 seconds
                            if f_n == 5*fps:
                                dynamic_x, dynamic_y, dynamic_w, dynamic_h = faces[0]

                            # Move the bounding box. Begin after 150 frames (5 seconds)
                            if f_n > 5*fps:
                                print('movement')
                                use_dynamic_box = True
                        '''
                        # save bounding box coordinates after 5 seconds
                        if f_n == 5 * fps:
                            dynamic_w = 150
                            dynamic_h = 150
                            dynamic_x = f_.shape[1] // 2 - dynamic_w // 2
                            dynamic_y = f_.shape[0] // 2 - dynamic_h // 2


                        # Move the bounding box. Begin after 150 frames (5 seconds)
                        if f_n > 5 * fps:
                            use_dynamic_box = True

                        if use_dynamic_box and dynamic_x is not None:
                            if start_time_bb is None:
                                start_time_bb = time.time()

                            elapsed_time = time.time() - start_time_bb
                            if elapsed_time < movement_duration:
                                # Move to the left until dynamic_x=0
                                dynamic_x -= int(movement_increment * dynamic_h)
                                if dynamic_x < 0:
                                    dynamic_x = 0
                            elif elapsed_time < 2 * movement_duration:
                                # Move to the center
                                dynamic_x += int(movement_increment * dynamic_h)
                                if dynamic_x > f_.shape[1] // 2 - dynamic_w // 2:
                                    dynamic_x = f_.shape[1] // 2 - dynamic_w // 2
                                # dynamic_y -= int(movement_increment * dynamic_h)
                            elif elapsed_time < 3 * movement_duration:
                                # Move to the top of the window
                                dynamic_y -= int(movement_increment * dynamic_h)
                                if dynamic_y <= f_.shape[0] // 8:
                                    dynamic_y = f_.shape[0] // 8
                            elif elapsed_time < 4 * movement_duration:
                                # Move to the bottom of the window
                                dynamic_y += int(movement_increment * dynamic_h)
                                if dynamic_y >= f_.shape[0] - dynamic_h - f_.shape[0] // 8:
                                    dynamic_y = f_.shape[0] - dynamic_h - f_.shape[0] // 8
                            elif elapsed_time < 5 * movement_duration:
                                # Return to the center
                                dynamic_y -= int(movement_increment * dynamic_h)
                                if dynamic_y <= f_.shape[0] // 2 - dynamic_h // 2:
                                    dynamic_y = f_.shape[0] // 2 - dynamic_h // 2
                            elif elapsed_time < 6 * movement_duration:
                                # Move to the right
                                dynamic_x += int(movement_increment * dynamic_h)
                                if dynamic_x >= f_.shape[1] - dynamic_w:
                                    dynamic_x = f_.shape[1] - dynamic_w
                            elif elapsed_time < 7 * movement_duration:
                                # Return to the center
                                dynamic_x -= int(movement_increment * dynamic_h)
                                if dynamic_x <= f_.shape[1] // 2 - dynamic_w // 2:
                                    dynamic_x = f_.shape[1] // 2 - dynamic_w // 2
                            else:
                                # Reset variables for the next movement sequence
                                start_time_bb = None
                                dynamic_x, dynamic_y = f_.shape[1] // 2 - dynamic_w // 2, f_.shape[
                                    0] // 2 - dynamic_h // 2

                        # Draw bounding box around the face
                        # for (x_rect, y_rect, w_rect, h_rect) in faces:
                        if use_dynamic_box and dynamic_x is not None:
                            x_rect, y_rect, w_rect, h_rect = dynamic_x, dynamic_y, dynamic_w, dynamic_h
                        else:
                            w_rect = 150
                            h_rect = 150
                            x_rect = f_.shape[1] // 2 - 150 // 2
                            y_rect = f_.shape[0] // 2 - 150 // 2

                        cv2.rectangle(f_, (x_rect, y_rect), (x_rect + w_rect, y_rect + h_rect), (0, 255, 0), 3)

                    if text2.get() == 'v11' or text2.get() == 'v12':
                        # Rotationsanzeige
                        rot_marker_path = 'c:/Philipp/Uni/_Master/Semester 9 (SoSe2023)/Master-Thesis/Probandentest/'
                        rot_marker_names = ['R1', 'R2', 'L1', 'L2', 'T1', 'T2', 'B1', 'B2']

                        if start_time_rot is None:
                            start_time_rot = time.time()
                            elapsed_time_rot = 0
                            update_interval_rot = 5  # Update interval in seconds
                            prev_first_letter = None

                        elapsed_time_rot = time.time() - start_time_rot
                        if elapsed_time_rot >= update_interval_rot:
                            # Choose a random rotational marker from the list
                            random_filename = random.choice(rot_marker_names)

                            while random_filename[0] == prev_first_letter:
                                random_filename = random.choice(rot_marker_names)
                            if random_filename[0] != prev_first_letter:
                                prev_first_letter = random_filename[0]

                            # Load the marker image
                            rot_img_path = rot_marker_path + random_filename + '.png'
                            rot_img = cv2.imread(rot_img_path)

                            # Display the loaded image (you can replace this with your processing logic)
                            # if rot_img is not None:
                            #     cv2.imshow('Anweisung zur Kopfrotation', cv2.resize(rot_img, (512, 512)))
                            start_time_rot = time.time()

                        if rot_img is not None:
                            # Resize the image to 100x100 pixels
                            resized_img = cv2.resize(rot_img, (100, 100))

                            # Get the dimensions of the resized image
                            height, width, _ = resized_img.shape

                            # Overlay the resized image onto the background image (top right corner)
                            f_[0:height, -width:] = resized_img

                    cv2.imshow('Logitech C930e', f_)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            frames = []
        finally:
            cap = None
            time.sleep(0.1)


def connect_sigma_cam():
    global frames_sigma, cap_sigma
    frames_sigma = []
    cap_sigma = None
    while alive:
        try:
            res_, fourcc_, fps_, cam_id_ = res, fourcc, fps, cam_id
            cam_id_ = cam_idx[~swap_camera_order]
            cap_sigma = cv2.VideoCapture(cam_id_, cv2.CAP_DSHOW)
            cap_sigma.set(5, fps)
            # cap_sigma.set(3, res[0])
            # cap_sigma.set(4, res[1])
            cap_sigma.set(3, frame_res_camera_1_sigma[0])
            cap_sigma.set(4, frame_res_camera_1_sigma[1])

            cap_sigma.set(6, cv2.VideoWriter.fourcc(*'YUY2'))
            w, h = cap_sigma.get(3), cap_sigma.get(4)
            # if h == 480:
            #     sel1.select()
            # elif h == 720:
            #     sel2.select()
            # elif h == 1080:
            #     sel3.select()
            fcc = cap_sigma.get(6)
            # if fcc == cv2.VideoWriter.fourcc(*'YUY2'):
            #     sel5.select()
            # elif fcc == cv2.VideoWriter.fourcc(*'MJPG'):
            #     sel4.select()
            r = ((640*480)/(w*h))**0.5
            # while (res_, fourcc_, fps_, cam_id_) == (res, fourcc, fps, cam_id) and alive:
            while (fourcc_, fps_) == (fourcc, fps) and alive:
                _, frame = cap_sigma.read()
                frame = cv2.flip(frame, 1)
                #print(cap_sigma.get(6)==cv2.VideoWriter.fourcc(*'YUY2'), time.time())
                #print(frame-cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420), cv2.COLOR_YUV2BGR_I420))
                frames_sigma.append((frame, time.time()))
                if recording and start_time:
                    while len(frames_sigma) > 100:
                        (f, t_) = frames_sigma.pop(0)
                        if dir_path and recording and time.time()<end_time:
                            with open(f'{dir_path}/missed_frames_sigma.csv', 'a') as csv:
                                csv.write(f'{t_}\n')
                else:
                    while len(frames_sigma) > 5:
                        frames_sigma.pop(0)
                f_ = cv2.resize(frame, (round(w*r), round(h*r)))
                if not start_time or not recording:
                    cv2.imshow('Sigma fp', f_)
                else:
                    t = time.time()-start_time
                    if int(t)%2:
                        cv2.putText(f_, f'{int(t)//60}:', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)
                    else:
                        cv2.putText(f_, f'{int(t)//60}', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)
                    cv2.putText(f_, f'{t%60:.1f}', (60, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 100, 200), 2)
                    cv2.imshow('Sigma fp', f_)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            frames_sigma = []
        finally:
            cap_sigma = None
            time.sleep(0.1)


def connect_cms50e():
    global bvp, hr, spo2
    bvp = []
    hr = []
    spo2 = []
    while alive:
        try:
            oximeter = CMS50Dplus(serial_port)
            datapoints = oximeter.get_realtime_data()
            for datapoint in datapoints:
                t = time.time()
                current_pulse_rate = datapoint.pulse_rate
                current_pulse_waveform = datapoint.pulse_waveform
                current_spO2 = datapoint.spO2

                _1 = (current_pulse_waveform, t)
                if not bvp or t>bvp[-1][-1]:
                    bvp.append(_1)
                if bvp and t==bvp[-1][-1]:
                    bvp[-1] = _1
                _2, _3 = (current_pulse_rate, t), (current_spO2, t)
                if not hr or t>hr[-1][-1]:
                    hr.append(_2)
                if not spo2 or t>spo2[-1][-1]:
                    spo2.append(_3)

                if len(bvp)>100:
                    bvp.pop(0)
                if len(hr)>100:
                    hr.pop(0)
                if len(spo2)>100:
                    spo2.pop(0)
        except Exception as e:
            # print(e)
            pass
        finally:
            time.sleep(0.1)
t1 = threading.Thread(target=connect_cms50e, daemon=True)
t2 = threading.Thread(target=connect_cam, daemon=True)
t1.start()
t2.start()
# if use_sigma:
#     t4 = threading.Thread(target=connect_sigma_cam, daemon=True)
#     t4.start()



import tkinter as tk
from tkinter import ttk
window = tk.Tk()
window.title('PHYS-recoder')
window.geometry('200x650+300+300')
window.resizable(False, False)
value_c = tk.StringVar()
c_list = ttk.Combobox(window, textvariable=value_c)
c_list['values'] = [i[1] for i in list_video_devices()]
c_list.current(0)
c_list.configure(state='readonly')
c_list.place(x=5, y=30)
lb1 = tk.Label(window, text='BVP:', font=('Times',10))
lb1.place(x=0, y=0)
lb2 = tk.Label(window, text='×', font=('Times',10), fg='red')
lb2.place(x=30, y=0)
lb3 = tk.Label(window, text='CAM:', font=('Times',10))
lb3.place(x=60, y=0)
lb4 = tk.Label(window, text='×', font=('Times',10), fg='red')
lb4.place(x=97, y=0)


diff = 25
lb5 = tk.Label(window, text='name', font=('Times',15))
lb5.place(x=10, y=30+diff)

if not os.path.exists(recordings_dir):
    os.mkdir(recordings_dir)

# get all recordings within recordings folder
directories = [d for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
# extract numbers from directory names and find the largest occuring number
numbers = [int(d[1:]) for d in directories if d.startswith('p') and d[1:].isdigit()]
largest_number = max(numbers) if numbers else 0

# increment the largest number and create a new directory name
next_folder_name = f"p{largest_number + 1:03d}"  # Format it to 'pXXX' where XXX is the incremented number

text1 = tk.Entry(window)
text1.insert(tk.INSERT, next_folder_name)
text1.place(x=5, y=60+diff)

lb6 = tk.Label(window, text='video', font=('Times',15))
lb6.place(x=10, y=80+diff)

text2 = tk.Entry(window)
text2.insert(tk.INSERT, 'v01')
text2.place(x=5, y=110+diff)

lb7 = tk.Label(window, text='duration', font=('Times',15))
lb7.place(x=10, y=130+diff)

text3 = tk.Entry(window)
text3.place(x=5, y=160+diff)
text3.insert(tk.INSERT, '60')

lb8 = tk.Label(window, text='REC:', font=('Times',10))
lb8.place(x=123, y=0)
lb9 = tk.Label(window, text='×', font=('Times',10), fg='red')
lb9.place(x=155, y=0)

lb10 = tk.Label(window, text='size', font=('Times',15))
lb10.place(x=0, y=180+diff)

v1 = tk.StringVar()
sel1 = tk.Radiobutton(window, text='480p', variable=v1, value='640x480')
sel1.place(x=5, y=200+diff)
sel2 = tk.Radiobutton(window, text='720p', variable=v1, value='1280x720')
sel2.place(x=65, y=200+diff)
sel3 = tk.Radiobutton(window, text='1080p', variable=v1, value='1920x1080')
sel3.place(x=125, y=200+diff)
sel1.select()

lb11 = tk.Label(window, text='camera codec', font=('Times',15))
lb11.place(x=0, y=220+diff)
v2 = tk.StringVar()
sel4 = tk.Radiobutton(window, text='MJPG', variable=v2, value='MJPG')
sel4.place(x=5, y=240+diff)
sel5 = tk.Radiobutton(window, text='YUY2', variable=v2, value='YUY2')
sel5.place(x=65, y=240+diff)
sel5.select()

lb12 = tk.Label(window, text='file codec', font=('Times',15))
lb12.place(x=0, y=260+diff)
ck1_v = tk.StringVar()
ck2_v = tk.StringVar()
ck3_v = tk.StringVar()
ck4_v = tk.StringVar()
ck5_v = tk.StringVar()
ck6_v = tk.StringVar()
ck1 = tk.Checkbutton(window, text='H264', onvalue='avc1', offvalue='', variable=ck1_v)
ck2 = tk.Checkbutton(window, text='I420', onvalue='I420', offvalue='', variable=ck2_v)
ck3 = tk.Checkbutton(window, text='RGBA', onvalue='RGBA', offvalue='', variable=ck3_v)
ck4 = tk.Checkbutton(window, text='MJPG', onvalue='MJPG', offvalue='', variable=ck4_v)
ck5 = tk.Checkbutton(window, text='PNGS', onvalue='PNGS', offvalue='', variable=ck5_v)
ck6 = tk.Checkbutton(window, text='FFV1', onvalue='FFV1', offvalue='', variable=ck6_v)
ck1.place(x=5, y=280+diff)
ck2.place(x=65, y=280+diff)
ck3.place(x=125, y=280+diff)
ck4.place(x=5, y=300+diff)
ck5.place(x=65, y=300+diff)
ck6.place(x=125, y=300+diff)
ck5.select()
ck3.select()

# Additional checkboxes for subject information
lb13 = tk.Label(window, text='___________________', font=('Times',15))
lb13.place(x=0, y=320+diff)
lb14 = tk.Label(window, text='Skin tone', font=('Times',15))
lb14.place(x=0, y=350+diff)
checked_skin_tone = tk.StringVar()
ck1_st_ = tk.Radiobutton(window, text='1', variable=checked_skin_tone, value='1')
ck2_st_ = tk.Radiobutton(window, text='2', variable=checked_skin_tone, value='2')
ck3_st_ = tk.Radiobutton(window, text='3', variable=checked_skin_tone, value='3')
ck4_st_ = tk.Radiobutton(window, text='4', variable=checked_skin_tone, value='4')
ck5_st_ = tk.Radiobutton(window, text='5', variable=checked_skin_tone, value='5')
ck6_st_ = tk.Radiobutton(window, text='6', variable=checked_skin_tone, value='6')
ck1_st_.place(x=5, y=370+diff)
ck2_st_.place(x=65, y=370+diff)
ck3_st_.place(x=125, y=370+diff)
ck4_st_.place(x=5, y=390+diff)
ck5_st_.place(x=65, y=390+diff)
ck6_st_.place(x=125, y=390+diff)
ck3_st_.select()


lb15 = tk.Label(window, text='Gender', font=('Times',15))
lb15.place(x=0, y=410+diff)
checked_gender = tk.StringVar()
ck1_gender_ = tk.Radiobutton(window, text='male', variable=checked_gender, value='1')
ck2_gender_ = tk.Radiobutton(window, text='female', variable=checked_gender, value='2')
ck1_gender_.place(x=65, y=413+diff)
ck2_gender_.place(x=125, y=413+diff)
ck1_gender_.select()

lb18 = tk.Label(window, text='Age', font=('Times',15))
lb18.place(x=0, y=440+diff)
T = tk.Text(window, height=1, width=30)
T.pack()
T.insert(tk.END, "28")
T.place(x=65, y=443+diff)

lb16 = tk.Label(window, text='Glasses', font=('Times',15))
lb16.place(x=0, y=470+diff)
checked_glasses = tk.StringVar()
ck1_glasses = tk.Checkbutton(window, text='', onvalue='1', offvalue='2', variable=checked_glasses)
ck1_glasses.place(x=65, y=473+diff)
checked_glasses.set(2)

lb16 = tk.Label(window, text='Hair Cover', font=('Times',15))
lb16.place(x=0, y=500+diff)
checked_hair_cover = tk.StringVar()
ck1_hair_cover = tk.Checkbutton(window, text='', onvalue='1', offvalue='2', variable=checked_hair_cover)
ck1_hair_cover.place(x=95, y=503+diff)
checked_hair_cover.set(2)

lb17 = tk.Label(window, text='Makeup', font=('Times',15))
lb17.place(x=0, y=530+diff)
checked_makeup = tk.StringVar()
ck1_makeup = tk.Checkbutton(window, text='', onvalue='1', offvalue='2', variable=checked_makeup)
ck1_makeup.place(x=65, y=533+diff)
checked_makeup.set(2)

lb18 = tk.Label(window, text='Sunlight illumination', font=('Times',15))
lb18.place(x=0, y=560+diff)
checked_sunlight = tk.StringVar()
ck1_sunlight = tk.Checkbutton(window, text='', onvalue='1', offvalue='2', variable=checked_sunlight)
ck1_sunlight.place(x=170, y=563+diff)
checked_sunlight.set(2)

recording = False
def b_record():
    # print(checked_skin_tone.get(), checked_gender.get(), checked_glasses.get(), checked_hair_cover.get(), checked_makeup.get())
    global recording, arduino_update_sent
    if not (frames and time.time()-frames[-1][-1]<0.2):
        recording = False
        b1.config(text='start')
        return
    recording = not recording
    arduino_update_sent = False
    b1.config(text='start' if not recording else 'stop')
    if recording:
        try:
            t3.join()
        except Exception:
            pass
        t3 = threading.Thread(target=record, daemon=True)
        t3.start()

b1 = tk.Button(window, text='start', command=b_record, width=15, height=1)
b1.place(x=40, y=595+diff)

def write(x):
    cv2.imwrite(x[1], x[0], [cv2.IMWRITE_PNG_COMPRESSION, 1])

def record():
    global bvp, hr, spo2, frames, frames_sigma, recording, start_time,end_time, f_n, dir_path
    f_n, dir_path = 0, ''
    start_time = 0
    bvp, hr, spo2, frames = bvp[-1:], hr[-1:], spo2[-1:], frames[-1:]

    if use_sigma:
        frames_sigma = frames_sigma[-1:]

    if not os.path.exists(recordings_dir + "/" + text1.get()):
        os.mkdir(recordings_dir + "/" + text1.get())
    dir_path_ = f'{recordings_dir}/{text1.get()}/{text2.get()}'
    if not os.path.exists(dir_path_):
        os.mkdir(dir_path_)
    else:
        dir_path_ = dir_path_ + "_" + datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        if not os.path.exists(dir_path_):
            os.mkdir(dir_path_)
    dir_path = dir_path_
    with open(f'{dir_path}/missed_frames.csv', 'w') as csv_missed_frames:
        csv_missed_frames.write('timestamp\n')
    if use_sigma:
        with open(f'{dir_path}/missed_frames_sigma.csv', 'w') as csv_missed_frames_sigma:
            csv_missed_frames_sigma.write('timestamp\n')
    with open(f'{dir_path}/info.txt', 'w') as txt:
        txt.write(f'cam model: {list_video_devices()[cam_idx[swap_camera_order]][1]}\n')
        fcc = cap.get(6)
        if fcc == cv2.VideoWriter.fourcc(*'YUY2'):
            txt.write('cam codec: YUY2\n')
        elif fcc == cv2.VideoWriter.fourcc(*'MJPG'):
            txt.write('cam codec: MJPG\n')
        else:
            txt.write('cam codec: UNKNOW\n')
        txt.write(f'size: {int(cap.get(3))}x{int(cap.get(4))}\n')
        date = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        txt.write(f'date: {date}')

        if use_sigma:
            txt.write(f'\n\ncam model 2: {list_video_devices()[cam_idx[~swap_camera_order]][1]}\n')
            fcc_sigma = cap_sigma.get(6)
            if fcc_sigma == cv2.VideoWriter.fourcc(*'YUY2'):
                txt.write('cam codec: YUY2\n')
            elif fcc_sigma == cv2.VideoWriter.fourcc(*'MJPG'):
                txt.write('cam codec: MJPG\n')
            else:
                txt.write('cam codec: UNKNOW\n')
            txt.write(f'size: {int(cap_sigma.get(3))}x{int(cap_sigma.get(4))}\n')
            date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            txt.write(f'date: {date}')
    with open(f'{dir_path}/subject_info.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["skin_tone","gender","age","glasses","hair_cover","makeup","sunlight_illumination"])
        csv_writer.writerow([checked_skin_tone.get(),checked_gender.get(),T.get("1.0",'end-1c'),checked_glasses.get(),checked_hair_cover.get(),checked_makeup.get(),checked_sunlight.get()])

    dur = float(text3.get())
    try:
    #video_outs = [cv2.VideoWriter(f'{dir_path}/video.avi' if i[0] !='a' else f'{dir_path}/video.mp4', cv2.VideoWriter.fourcc(*i), fps, res) for i in save_fourcc]
        video_outs = []
        video_outs_sigma = []
        writers = []
        writers_sigma = []
        tasks = []
        tasks_sigma = []
        png = False
        for i in save_fourcc:
            if i=='avc1':
                p = f'{dir_path}/video_ZIP_H264.avi'
            if i=='I420':
                p = f'{dir_path}/video_RAW_YUV420.avi'
            if i=='RGBA':
                p = f'{dir_path}/video_RAW_RGBA.avi'
            if i=='MJPG':
                p = f'{dir_path}/video_ZIP_MJPG.avi'
            if i=='FFV1':
                p = f'{dir_path}/video_ZIP_RAW_BGRA.avi'
            if i=='PNGS':
                png = f'{dir_path}/pictures_ZIP_RAW_RGB/'
                if not os.path.exists(png):
                    os.mkdir(png)
                if use_sigma:
                    png_sigma = f'{dir_path}/pictures_ZIP_RAW_RGB_sigma/'
                    if not os.path.exists(png_sigma):
                        os.mkdir(png_sigma)
                def png_write(p, n):
                    f = []
                    for i in p:
                        f.append((i[0], f'{png}{n:08}.png'))
                        n += 1
                    list(pool.map(write, f))
                def png_write_sigma(p, n):
                    f = []
                    for i in p:
                        f.append((i[0], f'{png_sigma}{n:08}.png'))
                        n += 1
                    list(pool.map(write, f))
                continue
            writer = cv2.VideoWriter(p, fourcc=cv2.VideoWriter.fourcc(*'RGBA'), fps=fps, frameSize=frame_res_camera_0_logitech)
            def f(x, writer=writer):
                global recording
                try:
                    for i in x:
                        writer.write(i)
                except Exception:
                    recording = False
            video_outs.append(f)
            writers.append(writer)

            if use_sigma:
                writer_sigma = cv2.VideoWriter(p.split('.avi')[0]+'_sigma.avi', fourcc=cv2.VideoWriter.fourcc(*'I420'), fps=fps, frameSize=frame_res_camera_1_sigma)

                def f_sigma(x, writer=writer_sigma):
                    global recording
                    try:
                        for i in x:
                            writer.write(i)
                    except Exception:
                        recording = False

                video_outs_sigma.append(f_sigma)
                writers_sigma.append(writer_sigma)
        if not save_fourcc:
            return
        with open(f'{dir_path}/BVP.csv', 'w') as f_bvp, open(f'{dir_path}/HR.csv', 'w') as f_hr, open(f'{dir_path}/SpO2.csv', 'w') as f_spo2,  open(f'{dir_path}/frames_timestamp.csv', 'w') as f_frames:
            f_bvp.write('timestamp,bvp\n')
            f_hr.write('timestamp,hr\n')
            f_spo2.write('timestamp,spo2\n')
            f_frames.write('frame,timestamp\n')
            start_time = time.time()
            end_time = float(dur)+start_time if int(dur) else 10**15
            while (frames[0][1]<end_time or not dur) and alive:
                t = time.time()
                if frames[-1][-1] >= end_time:
                    f = lambda x:x[-1]<end_time
                    bvp_, hr_, spo2_, frames_ = list(filter(f, bvp[:-1])), list(filter(f, hr[:-1])), list(filter(f, spo2[:-1])), list(filter(f, frames[:-1]))
                    if use_sigma:
                        frames_sigma_ =  list(filter(f, frames_sigma[:-1]))
                    recording = False
                else:
                    bvp_, hr_, spo2_, frames_ = bvp[:-1], hr[:-1], spo2[:-1], frames[:-1]
                    bvp, hr, spo2, frames = bvp[len(bvp_):], hr[len(hr_):], spo2[len(spo2_):], frames[len(frames_):]
                    if use_sigma:
                        frames_sigma_ = frames_sigma[:-1]
                        frames_sigma = frames_sigma[len(frames_sigma_):]
                for b,ts in bvp_:
                    f_bvp.write(f'{ts},{b}\n')
                for h,ts in hr_:
                    f_hr.write(f'{ts},{h}\n')
                for s,ts in spo2_:
                    f_spo2.write(f'{ts},{s}\n')
                f_ = [i[0] for i in frames_]
                if use_sigma:
                    f_sigma_ = [i[0] for i in frames_sigma_]

                for i in video_outs:
                    tasks.append(pool.submit(i, f_))
                if use_sigma:
                    for i in video_outs_sigma:
                        tasks_sigma.append(pool.submit(i, f_sigma_))
                if png:
                    png_write(frames_, f_n)
                    if use_sigma:
                        png_write_sigma(frames_sigma_, f_n)
                for _, ts in frames_:
                    f_frames.write(f'{f_n},{ts}\n')
                    f_n += 1
                for i in tasks:
                    i.result()
                tasks.clear()

                if use_sigma:
                    for i in tasks_sigma:
                        i.result()
                    tasks_sigma.clear()
                if not recording:
                    break
        time.sleep(max(t+0.1-time.time(), 0))
        recording = False
        for i in writers:
            i.release()
        if use_sigma:
            for i in writers_sigma:
                i.release()
    except Exception as e:
        print(e)
    finally:
        dir_path = ''
        recording = False
t3 = threading.Thread(target=record, daemon=True)

'''
# ToDo: Light, Motion, Excercise, Skin Color, Gender, Glasses, Hair Cover, Makeup, Age
lb15 = tk.Label(window, text='Light', font=('Times',15))
lb15.place(x=0, y=410+diff)
ck1_v_light = tk.StringVar()
ck2_v_light = tk.StringVar()
ck3_v_light = tk.StringVar()
ck1_light = tk.Checkbutton(window, text='Ceiling', onvalue='1', offvalue='', variable=ck1_v_light)
ck2_light = tk.Checkbutton(window, text='Additional', onvalue='1', offvalue='', variable=ck2_v_light)
ck3_light = tk.Checkbutton(window, text='Sunlight', onvalue='1', offvalue='', variable=ck3_v_light)
ck1_light.place(x=5, y=430+diff)
ck2_light.place(x=65, y=430+diff)
ck3_light.place(x=125, y=430+diff)
ck1_light.select()
ck3_light.select()

lb16 = tk.Label(window, text='Motion', font=('Times',15))
lb16.place(x=0, y=450+diff)
ck1_v_light = tk.StringVar()
ck2_v_light = tk.StringVar()
ck3_v_light = tk.StringVar()
ck1_light = tk.Checkbutton(window, text='Stationary', onvalue='1', offvalue='', variable=ck1_v_light)
ck2_light = tk.Checkbutton(window, text='Translation', onvalue='1', offvalue='', variable=ck2_v_light)
ck3_light = tk.Checkbutton(window, text='Rotation', onvalue='1', offvalue='', variable=ck3_v_light)
ck1_light.place(x=5, y=470+diff)
ck2_light.place(x=65, y=470+diff)
ck3_light.place(x=125, y=470+diff)
ck1_light.select()
ck3_light.select()
'''


def update_arduino_display():
    global recording, arduino_update_sent

    arduino = None

    try:
        arduino = serial.Serial(arduino_port, arduino_baudrate)
    except Exception as e:
        print(e)

    while arduino is not None:
        if recording:
            if not arduino_update_sent:
                arduino.write(0)
                arduino_update_sent = True
        else:
            arduino.write(1)
            arduino_update_sent = False
        '''
        if not arduino_update_sent:
            if recording:
                arduino.write(0)
                #arduino.write('XXXX'.encode())
                print("XXXX")
            else:
                subject_idx = text1.get()[2:]
                scenario_idx = text2.get()[1:]

                # if arduino_update_sent:
                # arduino.write(f'{subject_idx}.{scenario_idx}'.encode())
                arduino.write(1)
                print(f'{subject_idx}.{scenario_idx}')
            arduino_update_sent = True
            last_update_time = time.time()
        #else:
        #    if recording:
        #        arduino.write('XXXX'.encode())
        #        print("XXXX")
        if arduino_update_sent and time.time() - last_update_time >= 2:
            arduino_update_sent = False
        '''

# t5 = threading.Thread(target=update_arduino_display, daemon=True)
# t5.start()

def f_lb():
    global res, fourcc, fps, save_fourcc, cam_id
    global arduino_update_sent, arduino
    if bvp and time.time()-bvp[-1][-1]<0.2:
        lb2.config(text='√', font=('Times',10), fg='green')
    else:
        lb2.config(text='×', font=('Times',10), fg='red')
    if frames and time.time()-frames[-1][-1]<0.2:
        lb4.config(text='√', font=('Times',10), fg='green')
    else:
        lb4.config(text='×', font=('Times',10), fg='red')
    if recording:
        lb9.config(text='√', font=('Times',10), fg='green')
        # if not arduino_update_sent:
        #     arduino.write(0)
        #     arduino_update_sent = True
    else:
        lb9.config(text='×', font=('Times',10), fg='red')
        arduino_update_sent = False
        # if not arduino_update_sent:
        # arduino.write(1)
        #     arduino_update_sent = True

    video_devices = [i[1] for i in list_video_devices()]
    c_list['values'] = video_devices
    if video_devices:
        cam_id = video_devices.index(c_list.get())
    b1.config(text='start' if not recording else 'stop')
    res = [int(i) for i in v1.get().split('x')]
    fourcc = v2.get()
    # if fourcc == 'MJPG':
    #     fps = 30
    # elif fourcc == 'YUY2':
    #     if res == [1920, 1080]:
    #         fps = 5
    #     if res == [1280, 720]:
    #         fps = 10
    #     if res == [640, 480]:
    #         fps = 30
    save_fourcc = [i for i in (ck1_v.get(), ck2_v.get(), ck3_v.get(), ck4_v.get(), ck5_v.get(), ck6_v.get()) if i]
    window.after(100, f_lb)
f_lb()


if __name__ == '__main__':
    window.mainloop()
    alive = False
    try:
        t1.join()
        t2.join()
        # t4.join()
        t3.join()
        # t5.join()
    except Exception:
        pass

    try:
        pool.shutdown()
    except Exception:
        pass