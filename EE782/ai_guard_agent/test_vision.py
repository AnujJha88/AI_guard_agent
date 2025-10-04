from vision import load_trusted_faces, recognize_face

encodings, names = load_trusted_faces()
print("Trusted users:", names)

detected = recognize_face(encodings, names)
print("Detected:", detected)
