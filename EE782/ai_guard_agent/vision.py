# vision.py
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
from utils import save_intruder_face
# vision.py
FRAME_SKIP = 3
def load_trusted_faces(folder="trusted_faces"):
    encodings, names = [], []
    print("Loading trusted faces...")
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            encodings.append(enc[0])
            base_name = os.path.splitext(file)[0].split("_")[0]
            names.append(base_name)
            
    print(f"Loaded {len(names)} trusted faces: {list(set(names))}")
    return encodings, names


def recognize_face(known_encodings, known_names):
    cap = cv2.VideoCapture(0)
    detected_name = "Unknown"
    unknown_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for performance
        # (same as before, resized etc.)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, faces)

        unknown_count = 0
        detected_name = "Unknown"

        for enc, (top, right, bottom, left) in zip(encs, faces):
            distances = face_recognition.face_distance(known_encodings, enc)
            if len(distances) > 0 and min(distances) < 0.5:
                idx = distances.argmin()
                name = known_names[idx]
            else:
                name = "Unknown"
                unknown_count += 1

            detected_name = name

        cv2.imshow("GuardCam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if unknown_count > 1:
        return "Multiple_Intruders"
    return detected_name
