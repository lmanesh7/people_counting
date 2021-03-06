import streamlit as st
import pandas as pd
import pyrebase
import cv2
import imutils
import tempfile
import numpy as np
import datetime


from centroidtracker import CentroidTracker

from PIL import Image

config = {
    "apiKey": "AIzaSyBz9K9JuS4R6xIJrkVCA6kJ6BcuO_kx9aI",
    "authDomain": "test-a9823.firebaseapp.com",
    "databaseURL": "https://test-a9823-default-rtdb.firebaseio.com",
    "projectId": "test-a9823",
    "storageBucket": "test-a9823.appspot.com",
    "messagingSenderId": "1097818977650",
    "appId": "1:1097818977650:web:85b5d03d168fe3db7259eb",
    "measurementId": "G-KN10KC82JL"

}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

user = db.child("person counter").get()
us = user.val()
keys_list = []
vals_list = []
for i in us.values():
    # print(i)
    for j in i:
        keys_list.append(j)
        t = i[j]
        vals_list.append(t)
l = []
for i in keys_list:
    if i != "date":
        l.append(int(i))
opc = []
dat = []
for i in range(len(vals_list)):
    if i % 2 == 0:
        opc.append(vals_list[i])
    else:
        dat.append(vals_list[i])

df = pd.DataFrame({"lpc": l, "Time": dat, "Total": opc})

st.title("people counter")
st.write("No of persons in the present frame: %d" % (l[-1]))
st.write("No of persons in the total frames: %d" % (opc[-1]))
if st.checkbox("Show total db"):
    st.write(df)

if True:

    protopath = "MobileNetSSD_deploy.prototxt"
    modelpath = "MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


    def non_max_suppression_fast(boxes, overlapThresh):
        try:
            if len(boxes) == 0:
                return []

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last],
                                                       np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")
        except Exception as e:
            print("Exception occurred in non_max_suppression : {}".format(e))

upload = st.empty()
file = st.file_uploader("upload image", type=['jpg', 'png', 'jpeg'])

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
lpc_count = 0
opc_count = 0
object_id_list = []
if file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(file.read())

    upload.empty()
    while True:
        # ret,frame = cap.read()
        # frame = file.read()
        # frame = cv2.imread("peopl.jpg")
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        # frame = cv2.imdecode(file_bytes,1)
        # frame = cv2.imread(tfile.name)
        pil_image = Image.open(file).convert('RGB')
        frame = np.array(pil_image)
        # Convert RGB to BGR
        frame = frame[:, :, ::-1].copy()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        #objects = rects
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        #cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "Found: {}".format(lpc_count)
        opc_txt = "OPC: {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        #cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("people counter", frame)
        # write(fname, lpc_count, opc_count)
        st.image(frame)
        st.write("Total persons detected  :  %d" % lpc_count)
        st.write("If the persons are not detected then the image may not be clear or face may not be clear in the image")
        break
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
else:
    st.write("file not uploaded")
