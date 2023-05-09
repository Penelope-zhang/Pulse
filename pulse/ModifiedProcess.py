import cv2 as cv
import threading

import matplotlib.pyplot as plt
import numpy as np
import time
import traceback
from queue import Queue
import matplotlib.pyplot
from statsmodels.tsa.filters import hp_filter
from scipy import signal
from deprecated.sphinx import deprecated


'''
    The codes are implemented based on the following papers & Github projects:
    1.  https://github.com/thearn/webcam-pulse-detector
    2.  M. E. Wieler, T. G. Murphy, M. Blecherman, H. Mehta, and G. J. Bender, 
        “Infant heart-rate measurement and oxygen desaturation detection with a digital video camera using imaging photoplethysmography,” 
        J. Perinatol., 2021, doi: 10.1038/s41372-021-00967-1.
    3.  S. Sanyal and K. K. Nundy, “Algorithms for Monitoring Heart Rate and Respiratory Rate From the Video of a User’s Face,” 
        IEEE J. Transl. Eng. Heal. Med., vol. 6, no. May, pp. 1–11, 2018, doi: 10.1109/JTEHM.2018.2818687.
'''
class PulseDetect():

    # camera_id can be a video file or the camera id of your computer
    # data_size default = 30 fps * 10 s = 300 
    def __init__(self, camera_id=0, fathxml_path="haarcascade_frontalface_alt.xml",
                 eyexml_path="haarcascade_eye_tree_eyeglasses.xml", data_size=300, method="normal"):
        self.last_center = np.array([0, 0])
        self.camera = cv.VideoCapture(camera_id)
        self.face = np.array([0, 0, 0, 0])
        self.forehead = np.array([0, 0, 0, 0])
        self.eys = np.array([0, 0, 0, 0])
        self.face_haarcascade = cv.CascadeClassifier(fathxml_path)
        self.eys_haarcascade = cv.CascadeClassifier(eyexml_path)
        self.data_size = data_size
        self.frame = None
        self.frame_show = None
        self.bgr = None
        self.method = method
        self.times = []
        self.t0 = time.time()
        self.forehead_data = []
        self.freqs = []
        self.fft = []
        self.mean_values = []
        self.pruned_fft = []
        self.pruned_freqs = []
        self.time_points = []
        self.run_state = False
        self.bpms = []
        self.bpm_count = 1
        self.fps = int(self.camera.get(cv.CAP_PROP_FPS))
        self.total_frames = self.get_total_frames()
        self.step = 0
        self.time_gap = self.fps / self.total_frames
        self.landmark_detector = cv.face.createFacemarkLBF()
        self.landmark_detector.loadModel("lbfmodel.yaml")

    def read_video(self):
        return self.camera.read()

    def camera_opened(self):
        return self.camera.isOpened()

    def set_frame_local(self,frame):
        self.frame = frame
        self.frame_show = frame

    def initialize(self):
        self.times = []
        self.mean_values = []
        self.t0 = time.time()
        self.forehead_data = []
        self.freqs = []
        self.fft = []
        self.pruned_fft = []
        self.pruned_freqs = []
        self.time_points = []
        self.bpms = []
        self.bpm_count = 1
        self.step = 0

    def get_face_rect(self):
        faces = list(
            self.face_haarcascade.detectMultiScale(self.frame, minNeighbors=6, scaleFactor=1.2, minSize=(100, 100)))
        if len(faces) > 0:
            faces.sort(key=lambda x: x[-1] * x[-2])
            face = faces[-1]
            shifted = self.shift(face)
            if shifted > 10:
                self.face = face
        # self.get_eyes_rect()
        return self.face

    def get_total_frames(self):
        return int(self.camera.get(cv.CAP_PROP_FRAME_COUNT))

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shifted = np.linalg.norm(center - self.last_center)
        self.last_center = center
        return shifted

    def draw_rect(self, rect_area, color=(0, 255, 0)):
        x, y, w, h = rect_area
        cv.rectangle(self.frame_show, (x, y), (x + w, y + h), color=color, thickness=1)

    def get_forehead_rect(self):
        x, y, w, h = self.face
        newX = int(x + w * 0.5 - (w * 0.4 / 2))
        newY = int(y + h * 0.1 - (h * 0.1 / 2))
        newW = int(0.4 * w)
        newH = int(0.15 * h)
        self.forehead = np.array([newX, newY, newW, newH])

    def get_HSV(self):
        x, y, w, h = self.forehead
        try:
            forehead_hsv = cv.cvtColor(self.frame[y:y + h, x:x + w, :], cv.COLOR_BGR2HSV)
            HUE = (forehead_hsv[:, :, 0] / 360).reshape(-1, )
            new_HUE = HUE[np.where((HUE > 0) & (HUE < 0.1))]
            self.forehead_data.append(new_HUE)
            self.mean_values.append(np.mean(new_HUE))
            # cv.imshow("test", self.frame_show)
        except:
            self.times.pop()
            print("can't detect face")

    def get_face_cheeks(self):
        _, landmarks = self.landmark_detector.fit(self.frame, np.array([self.face]))
        landmark = landmarks[0][0]
        left_x1 = int(landmark[4][0])
        left_x2 = int(landmark[20][0])
        y1 = int((landmark[29][1] + landmark[30][1]) / 2)
        y2 = int((landmark[33][1] + landmark[50][1]) / 2)

        right_x1 = int(landmark[23][0])
        right_x2 = int(landmark[12][0])
        return left_x1, left_x2, right_x1, right_x2, y1, y2

    def get_cheek_HSV(self):
        left_x1, left_x2, right_x1, right_x2, y1, y2 = self.get_face_cheeks()
        try:
            left_cheek_hsv = cv.cvtColor(self.frame[y1:y2, left_x1:left_x2, :], cv.COLOR_BGR2HSV)
            right_cheek_hsv = cv.cvtColor(self.frame[y1:y2, right_x1:right_x2, :], cv.COLOR_BGR2HSV)
            left_HUE = (left_cheek_hsv[:, :, 0] / 360).reshape(-1, )
            right_HUE = (right_cheek_hsv[:, :, 0] / 360).reshape(-1, )
            new_left_HUE = left_HUE[np.where((left_HUE > 0) & (left_HUE < 0.1))]
            new_right_HUE = right_HUE[np.where((right_HUE > 0) & (right_HUE < 0.1))]
            new_HUE = np.concatenate((new_left_HUE, new_right_HUE))
            self.forehead_data.append(new_HUE)
            self.mean_vals.append(np.mean(new_HUE))

        except:
            self.times.pop()
            print("can't detect face")

    def use_HSV(self, length):

        # For using the IIR filter, the minimum length of data should be 123, given order of 20
        # Implementation of
        # Algorithms for Monitoring Heart Rate and Respiratory Rate From the Video of a User’s Face
        # Filtering is done before transforming, however according to the paper, it should be done after transforming.
        if (length <= 123):
            return 0
        fs = length / (self.times[-1] - self.times[0])
        interp = np.interp(self.time_points, self.times, self.mean_values)
        balanced_interp = interp - np.mean(np.hamming(length) * interp)
        sos = signal.iirfilter(N=20, Wn=[1.0, 2.0], fs=fs, output="sos")
        filtered = signal.sosfiltfilt(sos, balanced_interp)
        fft = np.abs(np.fft.rfft(filtered))
        bpm = self.freqs[np.argmax(fft)]
        return bpm

    # def draw_rect(self, rect_area, color=(0, 255, 0)):
    #     x, y, w, h = rect_area
    #     cv.rectangle(self.frame_show, (x, y), (x + w, y + h), color=color, thickness=1)


    def run(self):
        ret,frame = self.read_video()

        # Drop every two frames
        if self.step == 0 or self.step % 3 != 0:
            self.step  = self.step+1
            return

        # set frame
        self.set_frame_local(frame)

        # calculate time points and store
        t1 = self.t0 + self.step * self.time_gap
        self.times.append(t1)

        # Get forehead area
        self.get_face_rect()
        # self.get_forehead_rect()

        # Get HSV data
        # self.get_HSV()
        self.get_cheek_HSV()
        length = len(self.forehead_data)
        # self.draw_rect(self.face)
        # self.draw_rect(self.forehead)

        # only store the latest 10-second-long video's data
        if length > self.data_size:
            self.forehead_data = self.forehead_data[-self.data_size:]
            self.times = self.times[-self.data_size:]
            self.mean_values = self.mean_values[-self.data_size:]
            length = self.data_size

        # Once you have processed at least 1 second of data, do calculation
        if length > 30 :
            time_gap = self.times[-1] - self.times[0]
            fps = length / time_gap
            freqs = fps / length * np.arange(length // 2 + 1) * 60.
            self.freqs = freqs
            self.time_points = np.linspace(self.times[0], self.times[-1], length)
            bpm = self.use_HSV(length)
            if bpm == 0:
                return
            self.bpms.append(bpm)
            self.bpm_count = self.bpm_count + 1
            if len(self.bpms) > 64:
                self.bpms = self.bpms[-64:]
        #     x, y, w, h = self.face
        #     text = "(Your BPM is: %0.1f bpm. Please hold still)" % bpm
        #     cv.putText(self.frame_show, text, (x + w + 10, y - 10), cv.FONT_HERSHEY_PLAIN,
        #                    fontScale=1.2, color=(255, 0, 0), thickness=2)
        #     cv.putText(self.frame_show, "Current FPS: %0.1f fps" % fps, (0, 60), cv.FONT_HERSHEY_PLAIN, fontScale=1.2,
        #                color=(255, 0, 0), thickness=2)
        # cv.imshow("test", self.frame_show)
        # cv.waitKey(1)
        self.step  = self.step+1

    # Get the mean bpm of the last 10 frames
    def get_bpm(self):
        total_frames = self.get_total_frames()
        for i in range(total_frames):
            self.run()
        return np.average(self.bpms[-10:])
