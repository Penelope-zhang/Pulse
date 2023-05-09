import cv2

from app import times, face_haarcascade_alt, shift, forehead, forehead_data, mean_values, np, face, landmark_detector


# def get_HSV( frame ) -> object:
#     global forehead, mean_values
#     x, y, w, h = forehead
#     try:
#         forehead_hsv = cv2.cvtColor(frame[y:y + h, x:x + w, :], cv2.COLOR_BGR2HSV)
#         HUE = (forehead_hsv[:, :, 0] / 360).reshape(-1, )
#         new_HUE = HUE[np.where((HUE > 0) & (HUE < 0.1))]
#         forehead_data.append(new_HUE)
#         mean_values.append(np.mean(new_HUE))
#         return forehead_data
#     except:
#         times.pop()
#         print("can't detect face")

def get_forehead_rect():
    global forehead, face
    x, y, w, h = face
    newX = int(x + w * 0.5 - (w * 0.4 / 2))
    newY = int(y + h * 0.1 - (h * 0.1 / 2))
    newW = int(0.4 * w)
    newH = int(0.15 * h)
    forehead = np.array([newX, newY, newW, newH])
    return forehead


def get_face_rect(frame):
    global face
    faces = list(
        face_haarcascade_alt.detectMultiScale(frame, minNeighbors=4, scaleFactor=1.2, minSize=(100, 100)))
    if len(faces) > 0:
        faces.sort(key=lambda x: x[-1] * x[-2])
        face = faces[-1]
        shifted = shift(face)
        if shifted > 10:
            face = face
    return face


def get_face_cheeks(frame):
    global face
    _, landmarks = landmark_detector.fit(frame, np.array([face]))
    landmark = landmarks[0][0]
    left_x1 = int(landmark[4][0])
    left_x2 = int(landmark[20][0])
    y1 = int((landmark[29][1] + landmark[30][1]) / 2)
    y2 = int((landmark[33][1] + landmark[50][1]) / 2)
    right_x1 = int(landmark[23][0])
    right_x2 = int(landmark[12][0])
    return left_x1, left_x2, right_x1, right_x2, y1, y2


'''
fixed serious bug here caused by 'face = faces[-1]'
which means the face will always be the latest face, the method shift() will be be useless, 
and every face rect will differ greatly, resulting in extremely unstable bpms.
this bug is probably caused by merging or extracting the code from the main body.

Also the face rect shown in the page does not equal to the face rect used in the program,
see line 127 & 139 in app.py.
'''


def get_primary_face(faces_detected):
    global face
    faces = list(faces_detected)
    if len(faces) > 0:
        faces.sort(key=lambda x: x[-1] * x[-2])

        temp_face = faces[-1]
        shifted = shift(temp_face)
        # print(shifted)
        if shifted > 10:
            face = temp_face
    # print(face)
    return face
