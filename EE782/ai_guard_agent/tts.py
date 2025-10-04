# tts.py
import pyttsx3

engine = pyttsx3.init()

# Configure voices (check available with voices = engine.getProperty("voices"))
voices = engine.getProperty("voices")

# Default voice index (male/female based on system)
VOICE_INDEX = 0  
engine.setProperty("voice", voices[VOICE_INDEX].id)

def speak(text, rate=160, volume=1.0):
    """Speak with given speed and volume."""
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    engine.say(text)
    engine.runAndWait()

def set_voice(style="default"):
    """Switch between voice personalities"""
    if style == "calm":
        engine.setProperty("rate", 150)
    elif style == "angry":
        engine.setProperty("rate", 180)
        engine.setProperty("volume", 1.0)
    elif style == "friendly":
        engine.setProperty("rate", 165)
    else:
        engine.setProperty("rate", 160)
