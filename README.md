# WhatsApp Message Notification Router

An AI-powered, multimodal notification-routing agent for WhatsApp messages. For every incoming message, the system decides whether to:

- `notify` — interrupt the user immediately;
- `digest` — include the message in a later summary;
- `mute` — suppress low-value, repetitive, unwanted, suspicious, or unsafe content.

The router personalizes decisions using recipient behavior, group membership, business relationships, historical reactions, and notification load. It supports text messages, image posters/screenshots, and voice notes.

## Architecture

```text
Incoming message
  ├─ Text ──────────────────────────────────┐
  ├─ Image → Qwen2.5-VL visual extraction ──┼─> combined_text
  └─ Voice → Whisper transcription ─────────┘
                                                │
User / group / business context + history ─────┤
                                                v
NotificationRouterAgent
  ├─ retrieves relevant historical interactions
  ├─ applies deterministic safety guardrails
  ├─ uses Qwen2.5 for ambiguous decisions
  └─ selects supporting evidence IDs
                                                │
                                                v
output.csv: notify / digest / mute
```

## Key capabilities

- Multimodal extraction: Qwen2.5-VL reads posters/screenshots; Faster Whisper transcribes voice notes.
- Personalization: uses user notification behavior, group mute state, business opt-in/opt-out status, and historical opens, replies, dismissals, mutes, and reports.
- Safety guardrails: mutes scam-like messages, heavily forwarded chain spam, muted-group traffic, and opted-out promotions before LLM routing.
- Evidence-based decisions: includes relevant historical `message_id` values in every prediction where available.
- Reliable fallback: deterministic routing is used when the local LLM is unnecessary or unavailable.

## Project layout

```text
.
├── code/
│   └── main.py                 # Terminal entry point and routing pipeline
├── dataset/
│   ├── messages.csv            # Incoming messages to classify
│   ├── output.csv              # Generated submission file
│   ├── message_history.csv     # Historical messages
│   ├── message_events.csv      # Historical recipient reactions
│   ├── images.csv              # Image file mapping
│   ├── voice_notes.csv         # Voice-note file mapping
│   └── media/                  # Image and audio assets
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/)
- FFmpeg available on your `PATH` for MP3 transcription

Pull the local models:

```powershell
ollama pull qwen2.5:14b
ollama pull qwen2.5vl:7b
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

`requirements.txt` should include:

```text
pandas
ollama
faster-whisper
openpyxl
```

## Run

From the project root:

```powershell
python code/main.py
```

The pipeline reads all context from `dataset/` and writes the final submission to:

```text
dataset/output.csv
```

## Output contract

The generated CSV contains exactly one prediction per incoming message, using this column order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

Allowed actions: `notify`, `digest`, `mute`.

Allowed message types: `personal`, `urgent`, `event`, `payment`, `business_update`, `promotion`, `greeting`, `forward`, `spam`, `scam`, and `unknown`.

## Routing workflow

1. Validate and load dataset files.
2. Enrich each incoming message with user, group, business, and notification-load context.
3. Merge historical messages with user interaction events.
4. Aggregate past behavior by sender, group, and business.
5. Extract text from images and transcribe voice notes.
6. Build `combined_text` and detect urgency, payment, event, promotion, forwarding, and scam signals.
7. Route through `NotificationRouterAgent`:
   - retrieve relevant historical evidence;
   - apply safety and preference overrides;
   - ask Qwen2.5 only for ambiguous cases;
   - validate the decision and select evidence IDs.
8. Validate and write `dataset/output.csv`.

## Submission checklist

- [ ] `dataset/output.csv` has one row for each `dataset/messages.csv` row.
- [ ] Output columns are in the required order.
- [ ] `requirements.txt` includes every dependency.
- [ ] `code/main.py` runs from a terminal.
- [ ] Include this README and source code in `code.zip`.
- [ ] Include the required chat transcript log with the submission.
