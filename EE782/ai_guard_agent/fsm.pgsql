        ┌────────────────────┐
        │                    │
        │   IDLE (default)   │
        │                    │
        └───────┬────────────┘
                │ "Guard my room"
                ▼
        ┌────────────────────┐
        │                    │
        │       GUARD        │
        │                    │
        └───┬───────────┬────┘
            │ trusted    │ untrusted
            │ face       │ face detected
            ▼            ▼
  "Stop Guard"      ┌───────────────┐
     returns        │               │
     to IDLE        │  ESCALATION   │
                    │ (intruder)    │
                    │               │
                    └───────┬───────┘
                            │ resolved / stop guard
                            ▼
                          IDLE





IDLE (waiting)
   │
   ├── "Guard my room" → Guard Mode
   ▼
GUARD (monitoring faces)
   │
   ├── Trusted → Stay in Guard
   ├── "Stop guard" → Back to Idle
   └── Unknown → Escalation
   ▼
ESCALATION (dialogue + alarm if needed)
   │
   └── End → Back to Idle
