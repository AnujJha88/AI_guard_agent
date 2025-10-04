import cv2
import os

def enroll_user(name, save_dir="trusted_faces"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)

    print(f"📸 Press 'c' to capture {name}'s face, 'q' to quit.")
    count = 3
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            file_path = os.path.join(save_dir, f"{name}_{count}.jpeg")
            cv2.imwrite(file_path, frame)
            print(f"✅ Saved {file_path}")
            count += 1
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter the name of the person to enroll: ")
    enroll_user(username)