from transformers import pipeline
from tts import speak

PERSONALITY_MODE = "professional"

# Load small LLM (offline HuggingFace model)
chatbot = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

def guard_reply(user_input, level):
    mode = "polite security officer" if PERSONALITY_MODE == "professional" else "witty sarcastic AI guard"

    prompt = f"""
    You are an AI room guard with personality: {mode}.
    Escalation level: {level}.
    Intruder said: "{user_input}".
    Respond briefly but stay in character.
    """
    response = chatbot(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
    text = response[0]["generated_text"]
    guard_text = text.strip().split("\n")[-1]
    speak(guard_text)
    return guard_text


from deepface import DeepFace

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "neutral"