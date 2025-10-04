import speech_recognition as sr
from playsound import playsound
import os
def listen_for_command(timeout=5, phrase_time_limit=5):
    """
    Listen for a voice command and return it as text.
    - timeout: max seconds to wait for speech start
    - phrase_time_limit: max seconds to capture speech
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... (say something)")
        r.adjust_for_ambient_noise(source, duration=1)  # reduce background noise
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏳ No speech detected within timeout.")
            return ""
    try:
        # Using Google Web Speech API (free)
        command = r.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError:
        print("⚠️ Could not request results; check your internet connection.")
        return ""

def is_guard_command(text):
    """
    Check if the recognized text is the activation phrase.
    """
    return "guard my room" in text

def is_stop_command(text):
    return "stop guard" in text or "deactivate guard" in text
