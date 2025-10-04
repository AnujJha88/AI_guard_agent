packages = [ 
    "cv2", "mediapipe", "face_recognition", "deepface",
    "whisper", "speech_recognition", "vosk",
    "gtts", "pyttsx3", "TTS",
    "langchain", "google.generativeai", "openai",
    "transformers", "accelerate", "gradio", "streamlit",
    "pygame", "playsound", "pyaudio", "sounddevice", "torch"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} is installed")
    except ImportError as e:
        print(f"❌ {pkg} NOT installed ({e})")
