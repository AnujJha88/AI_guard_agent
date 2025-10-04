from asr import listen_for_command
from tts import speak,set_voice
from llm_agent import guard_reply
from utils import log_event
import os
from playsound import playsound

USE_LLM = True  # toggle between scripted vs LLM escalation

def escalate(level):
    if USE_LLM:
        # Ask intruder a question using LLM
        if level == 1:
            set_voice("friendly")
            speak("Who are you? This is a private room.")
            reply = listen_for_command(timeout=5, phrase_time_limit=5)

            if not reply:
                speak("Silence detected. Identifying as intruder.")
                log_event("Silent intruder at Level 1")
                return None
            guard_reply(reply, level)

        elif level == 2:
            set_voice("calm")
            speak("You are not recognized. Why are you here?")
            reply = listen_for_command(timeout=5, phrase_time_limit=5)

            if "sorry" in reply or "leaving" in reply:
                set_voice("friendly")
                speak("Good. Please leave now.")
                log_event("Intruder apologized at Level 2")
                return "intruder_backing_off"
            guard_reply(reply, level)

        elif level == 3:
            set_voice("angry")
            speak("Final warning! Security has been notified.")
            if os.path.exists("alarm.mp3"):
                playsound("alarm.mp3")
            log_event("Max escalation reached (Level 3)")
            return "escalated_to_max"

        return None
    else: 
        """
        Escalation logic with robustness and logging.
        """
        if level == 1:
            speak("Who are you? This is a private room.")
            reply = listen_for_command(timeout=5, phrase_time_limit=5)

            if not reply:
                speak("I did not hear a response. Identifying you as intruder.")
                log_event("Intruder silent at Level 1")
                return None

            if "friend" in reply or "roommate" in reply:
                speak("You are not enrolled as trusted. Please leave.")
                log_event("Intruder claimed 'friend/roommate'")
                return "untrusted_reply"

        elif level == 2:
            speak("You are not recognized. Please leave immediately.")
            reply = listen_for_command(timeout=5, phrase_time_limit=5)

            if not reply:
                speak("Still no response. You are trespassing.")
                log_event("Intruder silent at Level 2")
                return None

            if "sorry" in reply or "leaving" in reply:
                speak("Good. Please leave right now.")
                log_event("Intruder backed off at Level 2")
                return "intruder_backing_off"

        elif level == 3:
            speak("Final warning! Security has been notified.")
            # if os.path.exists("alarm.mp3"):
            #     playsound("alarm.mp3")
            log_event("Maximum escalation reached. Alarm triggered.")
            return "escalated_to_max"

    return None
