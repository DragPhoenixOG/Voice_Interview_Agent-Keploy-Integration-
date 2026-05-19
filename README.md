# 🎙️ AI Voice Interview Agent — Offline Edition

A fully local, fully free AI-powered mock interview system.
**No internet required at runtime. No API keys. No costs.**

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM (interview logic) | **Ollama** — Llama 3 / Mistral / Phi-3 | Free |
| Speech-to-Text | **Faster-Whisper** (local) | Free |
| Text-to-Speech | **pyttsx3** (system voice, offline) | Free |
| Backend | **FastAPI** | Free |
| Frontend | **Streamlit** | Free |

---

## ✨ Features

- 📄 **Resume parsing** — PDF & DOCX, extracted by Llama 3 locally (Since Ollama was taking too much space in RAM to parse the resume,I have used PyPDF2 locally and regex extraction)
- 🤖 **Personalized questions** — tailored to your actual resume skills & projects
- 🎙️ **Voice input** — microphone with automatic silence detection
- 👂 **Local STT** — Faster-Whisper runs on your CPU, no cloud
- 🔊 **Offline TTS** — pyttsx3 uses your OS's built-in voice engine
- 🧠 **Follow-up logic** — AI probes shallow answers for deeper detail
- 💬 **Multi-turn memory** — full conversation context maintained
- 📊 **Comprehensive feedback** — scores, strengths, weaknesses, action tips
- 📝 **Transcript logging** — saved as JSON in `transcripts/`

---

## 🏗️ Project Structure

```
voice-interview-agent/
├── app.py                       # FastAPI backend
├── start.py                     # One-command launcher
│
├── frontend/
│   └── streamlit_app.py         # Streamlit web UI
│
├── agents/
│   ├── greeter_agent.py         # Opening greeting via Ollama
│   ├── interviewer_agent.py     # Question generation + follow-up
│   └── feedback_agent.py        # End-of-interview report
│
├── audio/
│   ├── recorder.py              # Microphone recording
│   ├── stt.py                   # Faster-Whisper transcription
│   └── tts.py                   # pyttsx3 text-to-speech
│
├── resume/
│   └── parser.py                # PDF/DOCX parsing + Ollama extraction
│
├── memory/
│   └── transcript_store.py      # Session state + JSON persistence
│
├── utils/
│   ├── ollama_client.py         # Shared Ollama HTTP client
│   └── prompts.py               # All AI prompt templates
│
├── uploads/                     # Uploaded resume files
├── transcripts/                 # Saved interview transcripts (JSON)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Install Ollama

Download from **https://ollama.com/download** and install for your OS.

Then pull a model (choose one):
```bash
ollama pull llama3          # Recommended — 4.7 GB
ollama pull llama3:8b       # Lighter — 4.7 GB (same, just explicit tag)
ollama pull mistral         # Great alternative — 4.1 GB
ollama pull phi3            # Lightweight — 2.2 GB (faster on weak hardware)
```

Start the Ollama server (it may start automatically on install):
```bash
ollama serve
```

---

### Step 2 — Set Up Python Environment

```bash
# Clone / unzip the project
cd voice-interview-agent

# Create virtual environment
python -m venv venv

# Activate it:
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **macOS** — if you get PortAudio errors:
> ```bash
> brew install portaudio
> pip install sounddevice
> ```

> **Linux** — if audio doesn't work:
> ```bash
> sudo apt-get install espeak espeak-data libportaudio2 python3-dev
> ```

> **Windows** — pyttsx3 uses SAPI5 (built-in), should work out of the box.

---

### Step 4 — Configure

```bash
cp .env.example .env
```

Open `.env` and set your preferred model (no API key needed):
```env
OLLAMA_MODEL=llama3     # or mistral, phi3, llama3:8b
```

---

### Step 5 — Run

**Option A — One command (recommended):**
```bash
python start.py
```

**Option B — Two terminals:**
```bash
# Terminal 1: backend
python app.py

# Terminal 2: frontend
streamlit run frontend/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## ⚙️ Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3` | Local model to use for all AI tasks |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `WHISPER_MODEL` | `base` | STT model size (`tiny`/`base`/`small`/`medium`) |
| `MAX_QUESTIONS` | `8` | Number of interview questions |
| `TTS_RATE` | `165` | Speech rate (words per minute) |
| `TTS_VOLUME` | `1.0` | TTS volume (0.0–1.0) |
| `TTS_VOICE_INDEX` | `0` | System voice index |

**List available TTS voices:**
```bash
python -c "
import pyttsx3
e = pyttsx3.init()
for i, v in enumerate(e.getProperty('voices')):
    print(i, v.name)
"
```

---

## 🔧 Troubleshooting

### "Ollama not running"
```bash
ollama serve          # Start the server
ollama pull llama3    # Pull the model if not done yet
ollama list           # Check what's installed
```

### "Model not found"
The model name in `.env` must match exactly what `ollama list` shows.
E.g. if `ollama list` shows `llama3:latest`, set `OLLAMA_MODEL=llama3:latest`.

### Slow responses
Ollama runs on CPU by default. First response is slower (model loading).
Use a smaller model for faster inference:
```bash
ollama pull phi3          # 2.2 GB — fastest
ollama pull llama3:8b     # 4.7 GB — good balance
```

### Whisper model download
Faster-Whisper downloads its model on first use (~150 MB for `base`).
Use `WHISPER_MODEL=tiny` in `.env` for near-instant loading (less accurate).

### Microphone not recording
- **macOS:** System Preferences → Privacy → Microphone → allow Terminal/Python
- **Linux:** `sudo apt-get install libportaudio2` then `pip install sounddevice`
- **Windows:** Check microphone permissions in Windows Settings → Privacy

### pyttsx3 / TTS silent
- **Linux:** `sudo apt-get install espeak espeak-data`
- Try changing `TTS_VOICE_INDEX=1` in `.env`

---

## 📁 Transcripts

After each interview a JSON file is saved:
```
transcripts/transcript_a1b2c3d4_20241215_143022.json
```

Contains the full conversation, resume summary, and feedback report.

---

## 🛠️ Customization

- **Change interview style** → edit `utils/prompts.py`
- **Add more questions** → increase `MAX_QUESTIONS` in `.env`
- **Use GPU** → in `audio/stt.py` change `device="cpu"` to `device="cuda"`
- **Different model per agent** → pass `model=` to `chat()` in each agent file
- **Add more agents** → create files in `agents/`, import in `app.py`

---

## 📄 License

MIT — free to use, modify, and share.
