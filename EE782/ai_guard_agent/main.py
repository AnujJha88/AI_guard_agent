# main.py
from asr import listen_for_command, is_guard_command, is_stop_command
from utils import GuardState
from vision import load_trusted_faces, recognize_face
from agent import escalate
from utils import log_event

def handle_idle(state):
    cmd = listen_for_command()
    if is_guard_command(cmd):
        log_event("Guard mode activated")
        print("🛡 Guard mode activated. Monitoring...")
        return GuardState.GUARD
    return state


def handle_guard(state, encodings, names):
    print("👀 Monitoring for faces... Press q to stop camera.")
    detected = recognize_face(encodings, names)

    if detected == "Unknown":
        log_event("Single unknown intruder detected")
        state = GuardState.ESCALATION
    elif detected == "Multiple_Intruders":
        log_event("⚠️ Multiple intruders detected – escalating aggressively")
        state = GuardState.ESCALATION
    else:
        log_event(f"Trusted user detected: {detected}")
        print(f"✅ Trusted user detected: {detected}")
        cmd = listen_for_command()
        if is_stop_command(cmd):
            log_event("Guard mode deactivated by user")
            print("🛑 Guard mode deactivated. Back to idle.")
            return GuardState.IDLE
    return state


def handle_escalation(state):
    from agent import escalate
    for level in [1, 2, 3]:
        result = escalate(level)
        if result:
            break
    log_event("Escalation finished. Returning to guard mode.")
    print("🛡 Returning to Guard Mode automatically...")
    return GuardState.GUARD


def main():
    state = GuardState.IDLE
    encodings, names = load_trusted_faces()

    while True:
        if state == GuardState.IDLE:
            state = handle_idle(state)
        elif state == GuardState.GUARD:
            state = handle_guard(state, encodings, names)
        elif state == GuardState.ESCALATION:
            state = handle_escalation(state)


if __name__ == "__main__":
    main()
