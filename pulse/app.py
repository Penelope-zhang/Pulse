from tempfile import TemporaryFile

import imageio as imageio

from services.models.user import UserData
import base64
import io
from concurrent.futures import ThreadPoolExecutor
import flask
from flask import Flask, render_template, json, jsonify, make_response, session, request, copy_current_request_context
from json import dumps
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import load_model
import time
from datetime import datetime
import random
from numpy import ndarray, asarray
from scipy import signal
from threading import Lock
from flask_socketio import SocketIO, emit
import logging
from PIL import Image
import cv2

from services.utilities import face_detection

face_detection = face_detection

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, logger=True, engineio_logger=True, debug=True)
thread = None
thread_lock = Lock()
user = UserData

# # DOCS https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor
executor = ThreadPoolExecutor(2)

# # List of fields needs to be refactored
bpm_count = 1
bpm_history = []
bpms = []
current_time = datetime.now().strftime("%H:%M:%S")
# change data size to 256.
data_size = 256
eys = np.array([0, 0, 0, 0])
face = np.array([0, 0, 0, 0])
face_haarcascade_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_haarcascade_default = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml")
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")
fft = []
forehead = np.array([0, 0, 0, 0])
forehead_data = []
frequency = []
last_center = np.array([0, 0])
latest_bpm = []
mean_values = []
pruned_fft = []
pruned_freqs = []
run_state = False
t0 = time.time()
time_points = []
times = []
stored_frame = None
draw_debug = True
faces_detected = []
stored_bpms = []
stored_timestamps = []

# # for CORS
# @app.after_request
# def after_request( response ):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     response.headers.add('Access-Control-Allow-Headers',
#                          'Content-Type,Authorization')
#     # Put any other methods you need here
#     response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
#     return response


# # Flask api allows applications to stream large data efficiently
# # partitioned into smaller chunks, over potentially
# # long period of time. Ideal for video processing
# # https://blog.miguelgrinberg.com/post/video-streaming-with-flask

# # Multi-part processing is the key, allowing a chunk in the
# # video stream be an image, allowing an prior ML
# # processing before actor is presented results

# webCam = cv2.VideoCapture(0)

# # References :https://www.youtube.com/watch?v=mzX5oqd3pKA&t=524s
# # Reading all the models
gendermodel = load_model(r'gender_model.h5')
agemodel = load_model(r'age_model.h5')
emotionmodel = load_model(r'Emotion_model.h5')


# # Generate frames for live video feed
# def generate_frames():
#     frame_number = 0
#     # Times is used as an array of time keys
# global times
# global data_size
# global forehead
# global forehead_data
# global mean_values
# global frequency
# global time_points
# global bpm_count
# global bpms
# global bpm_history
# global bpm
# global stored_frame
# global faces_detected

# while True:
#     frame_number += 1
#     # read the camera frame
#     success, frame = webCam.read()
#     stored_frame = frame
#     # time appended to times array for data association in loop
#     # clear faces array
#     faces_detected = []
#     # try haarcascade_alt
#     try:
#         faces_detected = face_haarcascade_alt.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.2,
#                                                                minSize=(100, 100))
#         # Draw the rectangle around each face
#         if draw_debug:
#             for (x, y, w, h) in faces_detected:  # checking for multiple faces
#                 # CV2 face_haarcascade_alt uses BGR -Blue color rectangle
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
#     except:
#         print('face_haarcascade_alt unable to detect face.')
#
#     # if haarcascade_alt can't find faces try haarcascade_default
#     if len(faces_detected) <= 0:
#         try:
#             faces_detected = face_haarcascade_default.detectMultiScale(frame, 1.2, 5)
#             # Draw the rectangle around each face
#             if draw_debug:
#                 for (x, y, w, h) in faces_detected:  # checking for multiple faces
#                     # CV2 face_haarcascade_default uses BGR -Green color rectangle
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#         except:
#             print('face_haarcascade_default unable to detect face.')
#
#     # process every second frame only, 8FPS etc
#     if frame_number % 2:
#         # move time adding step here should solve the bug
#         t1 = time.time() - t0
#         times.append(t1)
#         executor.submit(bmp_async_calculation(frame, faces_detected))
#
#     ret, buffer = cv2.imencode('.jpg', frame)
#     frame = buffer.tobytes()
#
#     # Return the frames to video feed function
#     yield (b'--frame\r\n'
#            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Generate frames for live video feed
def process_frame():
    frame_number = 0
    # Times is used as an array of time keys
    global times
    global data_size
    global forehead
    global forehead_data
    global mean_values
    global frequency
    global time_points
    global bpm_count
    global bpms
    global bpm_history
    global bpm
    global stored_frame
    global faces_detected

    frame_number += 1
    # read the camera frame
    # time appended to times array for data association in loop
    # clear faces array
    faces_detected = []
    # try haarcascade_alt
    try:
        faces_detected = face_haarcascade_alt.detectMultiScale(stored_frame, minNeighbors=4, scaleFactor=1.2,
                                                               minSize=(100, 100))
        # Draw the rectangle around each face
        if draw_debug:
            for (x, y, w, h) in faces_detected:  # checking for multiple faces
                # CV2 face_haarcascade_alt uses BGR -Blue color rectangle
                cv2.rectangle(stored_frame, (x, y),
                              (x + w, y + h), (255, 0, 0), 1)
    except:
        print('face_haarcascade_alt unable to detect face.')

    # if haarcascade_alt can't find faces try haarcascade_default
    if len(faces_detected) <= 0:
        try:
            faces_detected = face_haarcascade_default.detectMultiScale(
                stored_frame, 1.2, 5)
            # Draw the rectangle around each face
            if draw_debug:
                for (x, y, w, h) in faces_detected:  # checking for multiple faces
                    # CV2 face_haarcascade_default uses BGR -Green color rectangle
                    cv2.rectangle(stored_frame, (x, y),
                                  (x + w, y + h), (0, 255, 0), 1)
        except:
            print('face_haarcascade_default unable to detect face.')

    # process every second frame only, 8FPS etc
    if frame_number % 2:
        # move time adding step here should solve the bug
        t1 = time.time() - t0
        times.append(t1)
        executor.submit(bmp_async_calculation(stored_frame, faces_detected))


def bmp_async_calculation(frame, faces_detected):
    global times
    global data_size
    global forehead
    global forehead_data
    global mean_values
    global frequency
    global time_points
    global bpm_count
    global bpms
    global bpm_history
    global bpm
    global latest_bpm
    global stored_bpms

    try:
        # face_detection.get_face_rect(frame)
        face_detection.get_primary_face(faces_detected)
    except:
        print('Face Rec Issue')

    try:
        forehead = face_detection.get_forehead_rect()
    except:
        print('Face forehead detection rec issue')

    # Combined forehead area with face cheeks
    try:
        left_x1, left_x2, right_x1, right_x2, y1, y2 = face_detection.get_face_cheeks(
            frame)
        x, y, w, h = forehead
        forehead_hsv = cv2.cvtColor(
            frame[y:y + h, x:x + w, :], cv2.COLOR_BGR2HSV)
        HUE = (forehead_hsv[:, :, 0] / 180).reshape(-1, )
        new_HUE = HUE[np.where((HUE > 0) & (HUE < 0.1))]
        left_cheek_hsv = cv2.cvtColor(
            frame[y1:y2, left_x1:left_x2, :], cv2.COLOR_BGR2HSV)
        right_cheek_hsv = cv2.cvtColor(
            frame[y1:y2, right_x1:right_x2, :], cv2.COLOR_BGR2HSV)
        left_HUE = (left_cheek_hsv[:, :, 0] / 180).reshape(-1, )
        right_HUE = (right_cheek_hsv[:, :, 0] / 180).reshape(-1, )
        new_left_HUE = left_HUE[np.where((left_HUE > 0) & (left_HUE < 0.1))]
        new_right_HUE = right_HUE[np.where(
            (right_HUE > 0) & (right_HUE < 0.1))]
        new_cheek_forehead_HUE = np.concatenate(
            (new_left_HUE, new_right_HUE, new_HUE))
        forehead_data.append(new_cheek_forehead_HUE)
        mean_values.append(np.mean(new_cheek_forehead_HUE))
    except:
        times.pop()
        print("can't detect face")

    length = len(forehead_data)

    try:
        if length > data_size:
            # [- var :] acts as a slice shorthand for pointer at X pos
            forehead_data = forehead_data[-data_size:]
            times = times[-data_size:]
            mean_values = mean_values[-data_size:]
            length = data_size
    except:
        print('Failure to obtain: instance one')

    try:
        # now wait until we have 256 frames
        if length >= 256:

            # since we process every two frames now,
            # we can assume that "originally" we have as much as twice the data amount
            # so the length is multiplied by 2, and we can have more time points.
            # After we get more time points, we can use linear interpolation to mock more the sample points then
            new_length = length * 2
            time_gap = times[-1] - times[0]
            fps = new_length / time_gap
            frequency = fps / new_length * np.arange(new_length // 2 + 1) * 60.
            time_points = np.linspace(times[0], times[-1], new_length)

            try:
                bpm = use_hsv(length)
            except:
                print('Failed to use HSV')

            if (bpm != 0):
                bpms.append(bpm)

            # narrow the window_range, now returns the last 1 second heartbeat
            window_range = int(fps)

            if len(bpms) > window_range:
                average_bpm = np.mean(bpms)
                bpms = bpms[-window_range:]
                stored_bpms.append(average_bpm)
                latest_bpm.append(average_bpm)
                stored_timestamps.append(time.time())

            if len(stored_bpms) > 300:
                stored_bpms = stored_bpms[-300:]

            global current_time

        # print("BPM calc performed.")
    except:
        print('BPM calc not performed')


# # if the person has bangs, we can use cheek area only, to detect whether a person has bangs,
# # we can use azure cognitive api

# def bmp_async_calculation_cheek_only( frame, faces_detected ):
#     global times
#     global data_size
#     global forehead
#     global forehead_data
#     global mean_values
#     global frequency
#     global time_points
#     global bpm_count
#     global bpms
#     global bpm_history
#     global bpm
#     global latest_bpm
#     global stored_bpms

#     try:
#         # face_detection.get_face_rect(frame)
#         face_detection.get_primary_face(faces_detected)
#     except:
#         print('Face Rec Issue')

#     try:
#         forehead = face_detection.get_forehead_rect()
#     except:
#         print('Face forehead detection rec issue')

#     # Combined forehead area with face cheeks
#     try:
#         left_x1, left_x2, right_x1, right_x2, y1, y2 = face_detection.get_face_cheeks(frame)
#         left_cheek_hsv = cv2.cvtColor(frame[y1:y2, left_x1:left_x2, :], cv2.COLOR_BGR2HSV)
#         right_cheek_hsv = cv2.cvtColor(frame[y1:y2, right_x1:right_x2, :], cv2.COLOR_BGR2HSV)
#         left_HUE = (left_cheek_hsv[:, :, 0] / 180).reshape(-1, )
#         right_HUE = (right_cheek_hsv[:, :, 0] / 180).reshape(-1, )
#         new_left_HUE = left_HUE[np.where((left_HUE > 0) & (left_HUE < 0.1))]
#         new_right_HUE = right_HUE[np.where((right_HUE > 0) & (right_HUE < 0.1))]
#         new_cheek_forehead_HUE = np.concatenate((new_left_HUE, new_right_HUE))
#         forehead_data.append(new_cheek_forehead_HUE)
#         mean_values.append(np.mean(new_cheek_forehead_HUE))
#     except:
#         times.pop()
#         print("can't detect face")

#     length = len(forehead_data)

#     try:
#         if length > data_size:
#             # [- var :] acts as a slice shorthand for pointer at X pos
#             forehead_data = forehead_data[-data_size:]
#             times = times[-data_size:]
#             mean_values = mean_values[-data_size:]
#             length = data_size
#     except:
#         print('Failure to obtain: instance one')

#     try:
#         if length >= 256:
#             new_length = length * 2
#             time_gap = times[-1] - times[0]
#             fps = new_length / time_gap
#             frequency = fps / new_length * np.arange(new_length // 2 + 1) * 60.
#             time_points = np.linspace(times[0], times[-1], new_length)

#             try:
#                 bpm = use_hsv(length)
#             except:
#                 print('Failed to use HSV')

#             if (bpm != 0):
#                 bpms.append(bpm)

#             # narrow the window_range, now returns the last 1 second heartbeat
#             window_range = int(fps)

#             if len(bpms) > window_range:
#                 average_bpm = np.mean(bpms)
#                 bpms = bpms[-window_range:]
#                 stored_bpms.append(average_bpm)
#                 latest_bpm.append(average_bpm)
#                 stored_timestamps.append(time.time())

#             if len(stored_bpms) > 300:
#                 stored_bpms = stored_bpms[-300:]

#             global current_time

#         # print("BPM calc performed.")
#     except:
#         print('BPM calc not performed')


# def frame_per_second_check( frame, i=None ):
#     # Handy hardware check for FPS
#     # i is to be initialised to 0 outside of while loop
#     i += 1
#     cv2.putText(frame, str(i), (10, 30),
#                 cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)


# def update_bpm_history( average_bpm, bpm_history ):
#     global current_time
#     # TODO: Extract, explain it's use.
#     try:
#         if (current_time != datetime.now().strftime("%H:%M:%S") and average_bpm > 1):
#             current_time = datetime.now().strftime("%H:%M:%S")
#             newBPM = {'date': datetime.now().isoformat(), 'value': average_bpm}
#             bpm_history.append(newBPM)
#             print(newBPM)
#     except:
#         print("Failure to append BPM to dict")


# # Function called to return frames to gender,emotion,age method
# # def gen_image():
# #     while True:
# #         success, frame = webCam.read()  # read the camera frame
# #         if success:
# #             return frame
# #         else:
# #             return

# For using the IIR filter, the minimum length of data should be 123, given order of 20 implementation of
# Algorithms for Monitoring Heart Rate and Respiratory Rate From the Video of a User’s Face
# Filtering is done before transforming, however according to the paper, it should be done after transforming.
def use_hsv( length ) -> object:
    if length <= 100:
        # print('lenght')
        # print(length)
        return 0

    # we assume we "originally" have twice the amount than now
    new_length = length * 2
    fs = new_length / (times[-1] - times[0])
    if not np.all(np.diff(times) > 0):
        print('times error')

    # Now we can generate more data points using linear interpolation.
    try:
        interp = np.interp(time_points, times, mean_values)
        balanced_interp = interp - np.mean(np.hamming(new_length) * interp)
    except:
        print('interpolation failed...')

    # change bpm range from 36~198 to 48~198 (0.6*60, 3.3*60)
    sos = signal.iirfilter(N=20, Wn=[0.8, 3.3], fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, balanced_interp)
    fft = np.abs(np.fft.rfft(filtered))
    bpm = frequency[np.argmax(fft)]
    return bpm


def shift(detected):
    global last_center
    x, y, w, h = detected
    center = np.array([x + 0.5 * w, y + 0.5 * h])
    shifted = np.linalg.norm(center - last_center)
    last_center = center
    return shifted


@app.route('/web_cam_feed')
def web_cam_feed():
    return flask.Response(
        stored_frame,
        # mimetype = 'multipart/x-mixed-replace; boundary=frame'
    )
    # return Flask.Response(generate_frames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')


@app.route('/privacy')
def privacy():
    # return flask.render_template('privacy.html')
    return render_template('privacy.html', async_mode=socketio.async_mode)


@app.route('/')
def index():
    # return flask.render_template('index.html')
    return render_template('index.html', async_mode=socketio.async_mode)


@app.route('/get-emotion')
def return_emotion_data():
    roi = region_of_interest_crop(stored_frame)

    if roi is None:
        return jsonify(Emotion=[])

    return jsonify(Emotion=get_emotions_from_roi(roi))


# Function to retrieve Emotion from Frame
def get_emotions_from_roi(roi):
    emotions = []
    try:
        roi_emotion = cv2.resize(roi, (48, 48))  # Resizing
        imagepixels1 = image.img_to_array(roi_emotion)
        imagepixels1 = np.expand_dims(imagepixels1, axis=0)
        imagepixels1 /= 255  # Scaling
    except:
        print('Unable to perform ROI')

    try:
        emotionpred = emotionmodel.predict(imagepixels1)  # Model prediction
        # Taking maximum value out of emotions
        max_index: ndarray[int] = np.argmax(emotionpred[0])
    except:
        print('Unable to perform emotion prediction')

    try:
        emotion = ('Angry', 'Disgusted', 'Fear', 'Happy',
                   'Neutral', 'Sad', 'Surprised')
        predicted_emotion = emotion[max_index]
    except:
        print('Emotional malfunction')
    print(predicted_emotion)
    emotions.append(predicted_emotion)  # Appending to an array
    return emotions


# API GET AGE
@app.route('/get-age')
def return_age_data():
    try:
        roi = region_of_interest_crop(stored_frame)
    except:
        roi = None
        print('Unable to detect face.')

    if roi is None:
        return jsonify(Age=[])

    return jsonify(Age=getAgesFromRoi(roi))


# Function to retrieve Age from Frame
def getAgesFromRoi(roi):
    try:
        ages = []
        # Resize to 200*200 as model is trained on this size
        roi = cv2.resize(roi, (200, 200))
        imagepixels = image.img_to_array(roi)
        imagepixels = np.expand_dims(imagepixels, axis=0)
        imagepixels /= 255  # Scaling

        img = imagepixels.reshape(-1, 200, 200, 3)

        age = agemodel.predict(img)  # Predicting age
        # print(int(age))
        age = int(age)
        ages.append(age)  # Appending age
    except:
        print('Age extraction issue')
    return ages


# API GET GENDER
@app.route('/get-gender')
def return_gender_data():
    global stored_frame
    roi = region_of_interest_crop(stored_frame)

    if roi is None:
        return jsonify(Gender=[])

    return jsonify(Gender=getGendersFromRoi(roi))


# Function to get Gender from frame
def getGendersFromRoi(roi):
    genders = []
    roi = cv2.resize(roi, (200, 200))  # Resizing image
    imagepixels = image.img_to_array(roi)
    imagepixels = np.expand_dims(imagepixels, axis=0)
    imagepixels /= 255  # Scaling the images

    predictions = gendermodel.predict(imagepixels)  # Predicting gender
    # If the predictions close to 0 then Male else Female
    if predictions < 0.3:
        genders.append('Male')
    else:
        genders.append('Female')
    return genders


# # To get Emotion,Age,Gender JSON together
# @app.route('/getAllData')
# def get_all_data():
#     frame = stored_frame
#     # print(frame)
#     # Detecting face in frame
#     roi = region_of_interest_crop(frame)

#     ages = []
#     emotions = []
#     genders = []

#     ages = getAgesFromRoi(roi)
#     emotions = get_emotions_from_roi(roi)
#     genders = getGendersFromRoi(roi)

#     # Return JSON
#     return jsonify(Age=ages, Emotion=emotions, Gender=genders)


def region_of_interest_crop(frame):
    global faces_detected

    # For each face detected on frame
    roi = None
    for (x, y, w, h) in faces_detected:
        # cropping region of interest i.e. face area from  image
        roi = frame[y:y + w, x:x + h]
    return roi


# Returns the current heartbeat
@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    global latest_bpm
    if len(mean_values) < 200 or (len(latest_bpm) <= 0):
        return "Not Enough Data!"
    else:
        latest_bpm = latest_bpm[-10:]
        result = {"bpm": [latest_bpm[-1]]}
        return jsonify(result)

# # Returns heartbeat history
# @app.route('/heartbeatHistory', methods=['Get'])
# def heartbeat_history():
#     global bpm_history
#     return make_response(dumps(bpm_history))


# # Toggle Debug
# @app.route('/toggleDebug')
# def toggle_debug():
#     global draw_debug
#     draw_debug = not draw_debug
#     return make_response(draw_debug)


# # Get the resting bpm based the bpms history
# @app.route('/getRestBpm')
# def get_rest_bpm():
#     global stored_bpms
#     if len(stored_timestamps) < 1:
#         return "No timestamp data!"
#     if int(stored_timestamps[-1] - stored_timestamps[0]) < 10:
#         return "Not enough data!"
#     heart_ranges = [[46, 50], [51, 55], [56, 60], [61, 65], [66, 70], [71, 75], [76, 80], [81, 85], [86, 90],
#                     [91, 95], [96, 100], [101, 105], [106, 110], [111, 150]]
#     df = pd.DataFrame(stored_bpms)
#     print(df[0].count())
#     rest_bpm_range = None
#     rest_range_count = 0
#     # find the heart range with the most counts
#     for heart_range in heart_ranges:
#         count = df[0].between(heart_range[0], heart_range[1]).sum()
#         if (count > rest_range_count):
#             rest_range_count = count
#             rest_bpm_range = heart_range
#     return jsonify(rest_bpm_range)


# # Get the lowest 20% bpm based on the bpms history
# @app.route('/getLowestRestBpm')
# def get_lowest_rest_bpm():
#     global stored_bpms
#     if len(stored_bpms) < 300:
#         return jsonify("Not enough data for calculating the rest heartrate")
#     return jsonify([np.average(np.sort(stored_bpms)[0:60])])


@socketio.event
def handle_frame(image):
    global stored_frame

    nparr = np.fromstring(image, np.uint8)
    stored_frame = cv2.imdecode(nparr, cv2.COLOR_HLS2RGB) # cv2.IMREAD_COLOR in OpenCV 3.1

    # Debugging for inspection of wrong media type 
    # print(type(stored_frame))
    # print(stored_frame)

    process_frame()

    try:
        # face_detection.get_face_rect(frame)
        face_detection.get_primary_face(faces_detected)
    except:
        print('Face Rec Issue')

    # emit the frame back
    emit('image_processed', image)


if __name__ == '__main__':
    socketio.run(app, debug=True)
