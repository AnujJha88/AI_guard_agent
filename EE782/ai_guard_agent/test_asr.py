from asr import listen_for_command, is_guard_command

cmd = listen_for_command()
if is_guard_command(cmd):
    print("✅ Guard mode activated")
else:
    print("❌ Not a guard command")
