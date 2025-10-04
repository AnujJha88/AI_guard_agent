class GuardState:
    IDLE = "idle"
    GUARD = "guard"
    ESCALATION = "escalation"

from datetime import datetime
import os
import cv2
def log_intruder(message, logfile="intruder_log.txt"):
    with open(logfile, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def save_intruder_face(frame, box, save_dir="unknown"):
    """
    Saves cropped intruder face instead of whole frame.
    :param frame: full video frame
    :param box: (top, right, bottom, left) bounding box
    :param save_dir: folder to save intruder images
    """
    os.makedirs(save_dir, exist_ok=True)
    top, right, bottom, left = box
    face_crop = frame[top:bottom, left:right]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"intruder_{timestamp}.jpg")
    cv2.imwrite(filename, face_crop)
    print(f"📷 Intruder face saved: {filename}")


def log_event(event, logfile="intruder_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"[{timestamp}] {event}\n")
    print(f"📝 Logged: {event}")