import numpy as np
import matplotlib; matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv2
import io
import time
from numpy import ndarray

webCam = cv2.VideoCapture(0)

heartbeat_count = 128
heartbeat_values = [0] * heartbeat_count
heartbeat_times = [time.time()] * heartbeat_count

# Matplotlib graph surface
fig = plt.figure()
assert isinstance(fig.add_subplot, object)
ax = fig.add_subplot(111)



while True:
    # FPS
    ret, frame = webCam.read()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # img = cv2.imread(frame)

    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_filter, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faces = frame[y:y + h, x:x + w]

    # crop_img = img[y:y + h, x:x + w]
    # Read the input image
    # img = cv2.imread(img)
    cv2.imshow('img', gray_filter)

    # Update the data
    heartbeat_values = heartbeat_values[1:] + [np.average(faces)]
    heartbeat_times = heartbeat_times[1:] + [time.time()]

    # Draw matplotlib graph to numpy array
    ax.plot(heartbeat_times, heartbeat_values)

    fig.canvas.draw()

    plot_img_np: ndarray = np.fromstring(fig.canvas.tostring_rgb(),
                                         dtype=np.uint8, sep='')

    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.cla()

    # Display the frames
    cv2.imshow('Crop', gray_filter)
    cv2.imshow('Graph', plot_img_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webCam.release()
cv2.destroyAllWindows()
