"""
frontend/streamlit_app.py
=========================
Streamlit frontend for the fully-offline Voice Interview Agent.
LLM: Ollama | STT: Faster-Whisper | TTS: pyttsx3
No internet connection needed once dependencies are installed.
"""

import os
import io
import json
import base64
import tempfile
import logging
import requests
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT    = os.getenv("API_PORT", "8000")
API_BASE    = f"http://{API_HOST}:{API_PORT}"
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Interview Agent — Offline",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #1e293b 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.page-header h1 { color: #a78bfa; font-size: 1.9rem; font-weight: 700; margin: 0; }
.page-header p  { color: #94a3b8; margin: 0.4rem 0 0; font-size: 0.95rem; }

/* Offline badge */
.offline-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    color: #6ee7b7; padding: 3px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    margin-top: 0.6rem; display: inline-block;
}

/* Status badges */
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-idle      { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }
.badge-active    { background: rgba(16,185,129,0.1);  color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.badge-recording { background: rgba(239,68,68,0.1);   color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); animation: blink 1.2s infinite; }
.badge-thinking  { background: rgba(139,92,246,0.1);  color: #c4b5fd; border: 1px solid rgba(139,92,246,0.3); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* Transcript */
.transcript-box {
    background: #0d1117; border: 1px solid #1e2633;
    border-radius: 12px; padding: 1.25rem;
    max-height: 420px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.84rem;
}
.msg-ai {
    background: rgba(139,92,246,0.07); border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem; margin: 0.4rem 0; color: #e2e8f0;
}
.msg-user {
    background: rgba(16,185,129,0.07); border-left: 3px solid #059669;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
    margin: 0.4rem 0 0.4rem 2.5rem; color: #e2e8f0;
}
.msg-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.1em; margin-bottom: 0.25rem; }
.msg-ai   .msg-label { color: #a78bfa; }
.msg-user .msg-label { color: #34d399; }

/* Feedback */
.fb-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 1.25rem; margin: 0.6rem 0;
}
.score-circle {
    font-size: 2.8rem; font-weight: 800; color: #a78bfa;
    text-align: center; line-height: 1;
}
.score-sub { text-align: center; color: #6b7280; font-size: 0.75rem;
             text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }

.strength-item  { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);
                  border-radius: 8px; padding: 0.6rem 0.9rem; margin: 0.35rem 0;
                  color: #a7f3d0; font-size: 0.88rem; }
.weakness-item  { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
                  border-radius: 8px; padding: 0.6rem 0.9rem; margin: 0.35rem 0;
                  color: #fde68a; font-size: 0.88rem; }
.improve-item   { background: rgba(139,92,246,0.08); border: 1px solid rgba(139,92,246,0.2);
                  border-radius: 8px; padding: 0.6rem 0.9rem; margin: 0.35rem 0;
                  color: #ddd6fe; font-size: 0.88rem; }
.recommend-box  {
    background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(16,185,129,0.08));
    border: 1px solid rgba(139,92,246,0.3); border-radius: 12px;
    padding: 1.25rem; text-align: center; margin: 1rem 0;
}

/* Skills pills */
.pill {
    display: inline-block; background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.25); color: #c4b5fd;
    padding: 2px 9px; border-radius: 12px; font-size: 0.76rem;
    font-weight: 500; margin: 2px;
}

/* Question card */
.question-card {
    background: rgba(139,92,246,0.06); border: 1px solid rgba(139,92,246,0.2);
    border-radius: 10px; padding: 1rem 1.25rem;
    color: #e2e8f0; font-size: 1rem; line-height: 1.6;
}
.q-counter {
    background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.2);
    border-radius: 8px; padding: 0.4rem 0.8rem; color: #a78bfa;
    font-size: 0.85rem; font-weight: 600; text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    transition: all 0.2s !important; padding: 0.55rem 1.4rem !important;
}
.stButton > button:hover { transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.45) !important; }

hr { border-color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
def _init():
    defaults = {
        "session_id":      None,
        "resume_data":     None,
        "stage":           "upload",   # upload|ready|in_progress|closing|completed
        "transcript":      [],
        "current_q":       None,
        "question_count":  0,
        "feedback":        None,
        "audio_status":    "idle",
        "candidate_name":  "Candidate",
        "ollama_ok":       None,
        "last_spoken_text": "",
        "pending_tts": None,    
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()



# ── API Helpers ───────────────────────────────────────────────────────────────
def api(method: str, endpoint: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{API_BASE}{endpoint}", timeout=180, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure `python app.py` is running in another terminal.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out — Ollama may be slow on first load. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return None


def add_msg(role: str, content: str):
    st.session_state.transcript.append({"role": role, "content": content})



def play_tts(text: str):

    if not text.strip():
        return

    # Prevent duplicate playback
    if st.session_state.last_spoken_text == text:
        return

    st.session_state.last_spoken_text = text

    try:

        response = requests.post(
            f"{API_BASE}/synthesize-speech",
            json={"text": text},
            timeout=60,
        )

        if response.status_code != 200:
            logger.warning("TTS request failed")
            return

        audio_bytes = response.content

        if not audio_bytes:
            logger.warning("No audio returned")
            return

        # Convert WAV → base64
        b64 = base64.b64encode(audio_bytes).decode()

        # Hidden autoplay audio
        audio_html = f"""
        <audio id="tts-audio" autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>

        <script>
        const audio = document.getElementById("tts-audio");
        audio.play().catch(err => console.log(err));
        </script>
        """

        st.markdown(audio_html, unsafe_allow_html=True)

    except Exception as e:
        logger.warning(f"TTS playback failed: {e}")


def record_mic() -> bytes | None:
    """Record from microphone with silence detection. Returns WAV bytes."""
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        st.error("❌ `sounddevice` / `scipy` not installed. Run: `pip install sounddevice scipy`")
        return None

    CHUNK           = int(SAMPLE_RATE * 0.1)   # 100 ms chunks
    MAX_CHUNKS      = int(45 / 0.1)            # 45s max
    SILENCE_TH      = 0.012
    SILENCE_CHUNKS  = 20                        # 2s silence → stop
    MIN_SPEECH      = 6                         # Must have 6 chunks of speech

    frames, silence_count, speech_frames = [], 0, 0
    status = st.empty()
    status.markdown('<div class="badge badge-recording">🔴 Recording — speak now</div>',
                    unsafe_allow_html=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as s:
            for _ in range(MAX_CHUNKS):
                chunk, _ = s.read(CHUNK)
                frames.append(chunk.copy())
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms > SILENCE_TH:
                    speech_frames += 1
                    silence_count  = 0
                elif speech_frames >= MIN_SPEECH:
                    silence_count += 1
                if speech_frames >= MIN_SPEECH and silence_count >= SILENCE_CHUNKS:
                    break
    except Exception as e:
        status.empty()
        st.error(f"❌ Recording failed: {e}")
        return None

    status.empty()

    if speech_frames < MIN_SPEECH:
        st.warning("⚠️ No speech detected. Please speak clearly and try again.")
        return None

    audio = np.concatenate(frames, axis=0)
    audio16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    from scipy.io.wavfile import write as wav_write
    wav_write(buf, SAMPLE_RATE, audio16)
    buf.seek(0)
    return buf.read()


# ── UI Sections ───────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="page-header">
        <h1>🎙️ AI Voice Interview Agent</h1>
        <p>Mock interviews powered by local AI — no internet needed at runtime</p>
        <div class="offline-badge">⚡ 100% Offline &nbsp;|&nbsp; Ollama + Faster-Whisper + pyttsx3</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📡 System Status")

        # Check Ollama connectivity
        if st.session_state.ollama_ok is None:
            res = api("get", "/health")
            st.session_state.ollama_ok = res and res.get("ollama") == "connected"

        ok = st.session_state.ollama_ok
        st.markdown(
            f'<div class="badge {"badge-active" if ok else "badge-recording"}">'
            f'{"✅ Ollama connected" if ok else "❌ Ollama not running"}</div>',
            unsafe_allow_html=True,
        )

        if not ok:
            st.caption("Start Ollama: `ollama serve`")
            st.caption(f"Pull model: `ollama pull {os.getenv('OLLAMA_MODEL','llama3')}`")
            if st.button("🔄 Retry connection"):
                st.session_state.ollama_ok = None
                st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Session")

        stage_labels = {
            "upload": ("Waiting for resume", "badge-idle"),
            "ready": ("Ready to start", "badge-active"),
            "in_progress": ("Interview in progress", "badge-active"),
            "closing": ("Wrapping up...", "badge-thinking"),
            "completed": ("Completed ✓", "badge-active"),
            "waiting_intro": ("Waiting for introduction", "badge-thinking"),
        }
        label, css = stage_labels.get(st.session_state.stage, ("Unknown", "badge-idle"))
        st.markdown(f'<div class="badge {css}">{label}</div>', unsafe_allow_html=True)

        if st.session_state.resume_data:
            rd = st.session_state.resume_data
            st.markdown(f"<br>**👤 {rd.get('name','Unknown')}**", unsafe_allow_html=True)
            skills = rd.get("skills", [])[:8]
            if skills:
                pills = "".join(f'<span class="pill">{s}</span>' for s in skills)
                st.markdown(pills, unsafe_allow_html=True)

        if st.session_state.question_count > 0:
            st.markdown(
                f'<div class="q-counter" style="margin-top:0.8rem">'
                f'Questions: {st.session_state.question_count}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**🛠 Stack:**")
        st.caption(f"🧠 LLM: `{os.getenv('OLLAMA_MODEL','llama3')}`")
        st.caption("🎤 STT: Faster-Whisper (local)")
        st.caption("🔊 TTS: pyttsx3 (offline)")

        st.markdown("---")
        if st.session_state.stage != "upload":
            if st.button("🔄 Reset Interview", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                _init()
                st.rerun()

        st.markdown("**📖 How to use:**")
        st.caption("1. Upload your resume")
        st.caption("2. Click Start Interview")
        st.caption("3. Listen to the AI question")
        st.caption("4. Click Record & speak")
        st.caption("5. Get detailed feedback")


def render_transcript():
    if not st.session_state.transcript:
        return
    st.markdown("### 📝 Conversation")
    html = '<div class="transcript-box">'
    for m in st.session_state.transcript:
        if m["role"] == "assistant":
            html += f'<div class="msg-ai"><div class="msg-label">🤖 Alex</div>{m["content"]}</div>'
        else:
            html += f'<div class="msg-user"><div class="msg-label">👤 You</div>{m["content"]}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_feedback(fb: dict):
    st.markdown("---")
    st.markdown("## 📊 Interview Feedback Report")

    # ── Score row ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, key, label in [
        (c1, "communication_feedback", "💬 Communication"),
        (c2, "overall_score",          "⭐ Overall"),
        (c3, "technical_feedback",     "💻 Technical"),
    ]:
        with col:
            val = fb.get(key, {})
            score = val.get("score") if isinstance(val, dict) else val
            st.markdown(
                f'<div class="fb-card">'
                f'<div class="score-circle">{score}<span style="font-size:1rem;color:#6b7280">/10</span></div>'
                f'<div class="score-sub">{label}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Recommendation ─────────────────────────────────────────────────────
    rec = fb.get("hiring_recommendation", "")
    st.markdown(
        f'<div class="recommend-box">'
        f'<div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.1em">Hiring Recommendation</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;margin-top:0.4rem">{rec}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Strengths / Weaknesses ─────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ✅ Strengths")
        for s in fb.get("strengths", []):
            st.markdown(f'<div class="strength-item">✓ {s}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("#### ⚠️ Weaknesses")
        for w in fb.get("weaknesses", []):
            st.markdown(f'<div class="weakness-item">⚡ {w}</div>', unsafe_allow_html=True)

    # ── Detailed tabs ──────────────────────────────────────────────────────
    st.markdown("#### 🔍 Detailed Analysis")
    t1, t2, t3 = st.tabs(["📢 Communication", "💻 Technical", "🤝 Behavioral"])
    for tab, key in [(t1, "communication_feedback"), (t2, "technical_feedback"), (t3, "behavioral_feedback")]:
        with tab:
            d = fb.get(key, {})
            st.markdown(f'<div class="fb-card">{d.get("feedback","N/A")}</div>', unsafe_allow_html=True)

    # ── Improvement tips ───────────────────────────────────────────────────
    st.markdown("#### 🎯 Areas to Improve")
    for tip in fb.get("areas_to_improve", []):
        st.markdown(f'<div class="improve-item">→ {tip}</div>', unsafe_allow_html=True)

    # ── Closing message ────────────────────────────────────────────────────
    closing = fb.get("closing_message", "")
    if closing:
        st.info(f"💌 {closing}")

    # ── Download ───────────────────────────────────────────────────────────
    if st.session_state.transcript:
        lines = []
        for m in st.session_state.transcript:
            speaker = "ALEX (AI)" if m["role"] == "assistant" else "YOU"
            lines.append(f"[{speaker}]\n{m['content']}\n")
        report = (
            "=== INTERVIEW TRANSCRIPT ===\n\n"
            + "\n".join(lines)
            + "\n=== FEEDBACK REPORT ===\n\n"
            + json.dumps(fb, indent=2)
        )
        st.download_button(
            "📥 Download Full Report",
            data=report,
            file_name=f"interview_report_{(st.session_state.session_id or 'local')[:8]}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── Main App Flow ─────────────────────────────────────────────────────────────
def main():
    render_header()
    render_sidebar()

    stage = st.session_state.stage
    # ─────────────────────────────────────────────────────
    # Deferred TTS Playback
    # ─────────────────────────────────────────────────────
    if st.session_state.pending_tts:

        text = st.session_state.pending_tts

        # Clear BEFORE playback
        st.session_state.pending_tts = None

        play_tts(text)

    # ═══ UPLOAD RESUME ════════════════════════════════════════════════════════
    if stage == "upload":
        st.markdown("### 📎 Step 1 — Upload Your Resume")
        st.caption("PDF or DOCX · Parsed locally with Ollama · Not sent anywhere")

        uploaded = st.file_uploader(
            "Drop your resume here",
            type=["pdf", "docx", "doc"],
            label_visibility="collapsed",
        )

        if uploaded:
            with st.spinner(f"🧠 Parsing with Ollama (`{os.getenv('OLLAMA_MODEL','llama3')}`) — may take 20-40s on first run..."):
                res = api(
                    "post", "/upload-resume",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )

            if res and res.get("success"):
                st.session_state.resume_data = res["resume_data"]
                st.session_state.stage = "ready"
                st.success(f"✅ Resume parsed! Welcome, **{res['resume_data'].get('name','Candidate')}**")

                rd = res["resume_data"]
                with st.expander("📋 Parsed Resume Preview", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Name:** {rd.get('name','N/A')}")
                        st.markdown(f"**Email:** {rd.get('email') or 'N/A'}")
                        if rd.get("education"):
                            st.markdown("**Education:**")
                            for e in rd["education"]:
                                st.caption(f"• {e}")
                    with col2:
                        if rd.get("skills"):
                            pills = "".join(f'<span class="pill">{s}</span>' for s in rd["skills"][:12])
                            st.markdown("**Skills:**")
                            st.markdown(pills, unsafe_allow_html=True)
                        if rd.get("experience"):
                            st.markdown("**Experience:**")
                            for ex in rd["experience"][:2]:
                                st.caption(f"• {str(ex)[:90]}")
                st.rerun()

    # ═══ READY TO START ═══════════════════════════════════════════════════════
    elif stage == "ready":
        rd = st.session_state.resume_data
        st.markdown(f"### ✅ Ready to interview **{rd.get('name', 'you')}**")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(
                "**🎙️ How the interview works:**\n\n"
                "1. Click **Start Interview** → AI greets you and asks the first question\n"
                "2. Click **🎙️ Record Answer** → speak your answer (auto-stops on silence)\n"
                "3. AI transcribes, thinks, and asks the next question\n"
                "4. After all questions → detailed feedback report with scores\n\n"
                # f"_Model: `{os.getenv('OLLAMA_MODEL','llama3')}` · STT: Faster-Whisper · TTS: pyttsx3_"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start Interview", use_container_width=True):
                with st.spinner("Initializing session..."):
                    res = api("post", "/start-interview", json=st.session_state.resume_data)

                if res and res.get("success"):
                    st.session_state.session_id     = res["session_id"]
                    st.session_state.candidate_name = res["candidate_name"]
                    greeting = res["greeting"]

                    add_msg("assistant", greeting)

                    # Move UI first
                    st.session_state.stage = "waiting_intro"

                    # Queue greeting TTS
                    st.session_state.pending_tts = greeting

                    st.rerun()
    # ═══ WAITING FOR INTRO ════════════════════════════════════════════════════
    elif stage == "waiting_intro":

        render_transcript()

        st.markdown("### 👋 Introduce Yourself")

        st.info("Please introduce yourself briefly before starting the interview.")

        if st.button("🎙️ Start Introduction", use_container_width=True):

            audio_bytes = record_mic()

            if audio_bytes:

                with st.spinner("Processing introduction..."):

                    res = api(
                        "post",
                        "/process-answer",
                        data={"session_id": st.session_state.session_id},
                        files={"audio_file": ("intro.wav", audio_bytes, "audio/wav")},
                    )

                if res:

                    intro = res.get("transcription", "")

                    if intro:
                        add_msg("user", intro)

                    # Backend already generated first question
                    next_q = res.get("next_response", "")

                    if next_q:

                        add_msg("assistant", next_q)

                        st.session_state.current_q = next_q
                        st.session_state.stage = "in_progress"

                        # Queue TTS
                        st.session_state.pending_tts = next_q

                        st.rerun()
                    

    # ═══ IN PROGRESS ══════════════════════════════════════════════════════════
    elif stage == "in_progress":
        st.markdown(f"### 🎙️ Interview — Question {st.session_state.question_count}")
        render_transcript()
        st.markdown("---")

        if st.session_state.current_q:
            st.markdown("**🤖 Current Question:**")
            st.markdown(f'<div class="question-card">{st.session_state.current_q}</div>',
                        unsafe_allow_html=True)

            col_play, col_gap = st.columns([1, 3])
            with col_play:
                # if st.button("🔊 Replay Question"):
                #     play_tts(st.session_state.current_q)
                if st.button("🔊 Replay Question"):

                    st.session_state.last_spoken_text = ""

                    st.session_state.pending_tts = st.session_state.current_q

                    st.rerun()

        st.markdown("---")
        st.markdown("**🎤 Your Turn:**")

        col_btn, col_status = st.columns([1, 2])
        with col_btn:
            record_btn = st.button("🎙️ Record Answer", use_container_width=True, type="primary")
        with col_status:
            s = st.session_state.audio_status
            badge = {
                "recording":  "badge-recording",
                "processing": "badge-thinking",
            }.get(s, "badge-idle")
            label = {"recording": "🔴 Recording...", "processing": "⚙️ Processing..."}.get(s, "⚫ Idle")
            st.markdown(f'<div class="badge {badge}">{label}</div>', unsafe_allow_html=True)

        if record_btn:
            st.session_state.audio_status = "recording"
            audio_bytes = record_mic()

            if audio_bytes:
                st.session_state.audio_status = "processing"
                with st.spinner("⚙️ Transcribing and thinking..."):
                    res = api(
                        "post", "/process-answer",
                        data={"session_id": st.session_state.session_id},
                        files={"audio_file": ("answer.wav", audio_bytes, "audio/wav")},
                    )
                st.session_state.audio_status = "idle"

                if res:
                    if not res.get("success"):
                        st.warning(res.get("message", "Could not understand. Please try again."))
                    else:
                        t = res.get("transcription", "")
                        if t:
                            st.success(f"✅ You said: *\"{t}\"*")
                            add_msg("user", t)

                        nr = res.get("next_response", "")
                        # if nr:
                        #     add_msg("assistant", nr)
                        #     play_tts(nr)
                        if nr:

                            add_msg("assistant", nr)

                            st.session_state.current_q = nr

                            st.session_state.pending_tts = nr

                            if res.get("next_action") == "generate_feedback":
                                st.session_state.stage = "closing"
                                st.session_state.current_q = None
                            else:
                                st.session_state.question_count = res.get(
                                    "question_number",
                                    st.session_state.question_count
                                )

                            st.rerun()

                        if res.get("next_action") == "generate_feedback":
                            st.session_state.stage = "closing"
                            st.session_state.current_q = None
                        else:
                            st.session_state.current_q      = nr
                            st.session_state.question_count = res.get("question_number", st.session_state.question_count)
                st.rerun()
            else:
                st.session_state.audio_status = "idle"

        st.markdown("---")
        if st.button("⏹️ End Interview & Get Feedback"):
            st.session_state.stage = "closing"
            st.rerun()

    # ═══ GENERATING FEEDBACK ══════════════════════════════════════════════════
    elif stage == "closing":
        st.markdown("### 🧠 Generating Your Feedback Report...")
        render_transcript()

        with st.spinner(f"Ollama is analyzing your interview (30-60s depending on model size)..."):
            res = api("post", "/generate-feedback", json={"session_id": st.session_state.session_id})

        if res and res.get("success"):
            st.session_state.feedback = res["feedback"]
            st.session_state.stage    = "completed"
            # Speak brief TTS summary
            summary = res.get("tts_summary", "")
            if summary:

                st.session_state.pending_tts = summary

                st.rerun()
        else:
            st.error("Feedback generation failed. Ensure Ollama is running and try again.")
            if st.button("Retry"):
                st.rerun()

    # ═══ COMPLETED ════════════════════════════════════════════════════════════
    elif stage == "completed":
        render_transcript()
        if st.session_state.feedback:
            render_feedback(st.session_state.feedback)
        st.success("✅ Interview complete! Transcript saved to the `transcripts/` folder.")


if __name__ == "__main__":
    main()
