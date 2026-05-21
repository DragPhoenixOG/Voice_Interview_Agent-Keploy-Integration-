"""
app.py
======
FastAPI backend for the Voice Interview Agent.
All AI runs locally via Ollama. TTS is offline via pyttsx3.
STT is local via Faster-Whisper.

Endpoints:
  GET  /health                  — liveness + Ollama connectivity check
  GET  /models                  — list locally available Ollama models
  POST /upload-resume           — parse uploaded PDF/DOCX
  POST /start-interview         — create session + get greeting
  POST /first-question          — generate the opening interview question
  POST /process-answer          — transcribe audio + return next question
  POST /generate-feedback       — produce final feedback report
  POST /synthesize-speech       — text → WAV bytes (pyttsx3)
  GET  /session/{id}            — current session state
  DELETE /session/{id}          — end and save session
"""

import os
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn
from dotenv import load_dotenv

from resume.parser import parse_resume
from memory.transcript_store import create_session, get_session, delete_session
from agents.greeter_agent import generate_greeting
from agents.interviewer_agent import (
    generate_question,
    generate_followup_question,
    generate_closing_statement,
    should_ask_followup,
    is_interview_complete,
)
from agents.feedback_agent import generate_feedback, format_feedback_for_tts
from audio.stt import transcribe_audio, is_meaningful_response
from audio.tts import get_tts_audio_bytes
from utils.ollama_client import list_local_models, _check_ollama_running

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Voice Interview Agent (Offline)",
    description="AI mock interviews powered by Ollama + Faster-Whisper + pyttsx3 — 100% local",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Check API is up and Ollama is reachable."""
    ollama_ok = _check_ollama_running()
    return {
        "status":      "ok" if ollama_ok else "degraded",
        "service":     "Voice Interview Agent",
        "ollama":      "connected" if ollama_ok else "not running — start with: ollama serve",
        "tts_engine":  "pyttsx3 (offline)",
        "stt_engine":  "faster-whisper (offline)",
    }


@app.get("/models")
async def list_models():
    """Return locally available Ollama models."""
    models = list_local_models()
    return {
        "models":        models,
        "active_model":  os.getenv("OLLAMA_MODEL", "qwen2:0.5b"),
    }


# ── Resume ────────────────────────────────────────────────────────────────────

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accept a PDF or DOCX resume, parse it with Ollama, return structured data.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".doc"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload PDF or DOCX.",
        )

    file_id   = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    try:
        content = await file.read()
        save_path.write_bytes(content)
        logger.info(f"Resume saved: {save_path}")

        resume_data = parse_resume(str(save_path))
        return {
            "success":     True,
            "resume_data": resume_data,
            "message":     f"Resume parsed for {resume_data.get('name', 'Unknown')}",
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process resume: {e}")


# ── Session Start ─────────────────────────────────────────────────────────────

@app.post("/start-interview")
async def start_interview(resume_data: dict):
    """
    Create a new session and generate the opening greeting.
    Body: the resume_data dict returned by /upload-resume
    """
    try:
        session = create_session()
        session.set_resume(resume_data)
        # Waiting for candidate introduction
        session.set_state("waiting_intro")

        greeting = generate_greeting(resume_data)
        session.add_message(
            role="assistant",
            content=greeting,
            is_question=False,
            metadata={"type": "greeting"},
        )

        logger.info(f"Session started: {session.session_id[:8]}")
        return {
            "success":        True,
            "session_id":     session.session_id,
            "greeting":       greeting,
            "candidate_name": resume_data.get("name", "Candidate"),
        }

    except Exception as e:
        logger.error(f"Start interview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── First Question ────────────────────────────────────────────────────────────

@app.post("/first-question")
async def first_question(body: dict):
    """Generate and return the first interview question."""
    session = get_session(body.get("session_id", ""))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        q = generate_question(
            resume_data=session.resume_data,
            conversation_history=session.get_conversation_history(),
            asked_questions=session.get_asked_questions(),
        )
        session.add_message(
            role="assistant",
            content=q,
            is_question=True,
            metadata={"type": "question", "question_number": 1},
        )
        session.set_state("in_progress")

        return {
            "success":         True,
            "question":        q,
            "question_number": session.question_count,
        }

    except Exception as e:
        logger.error(f"First question failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Process Answer ────────────────────────────────────────────────────────────

@app.post("/process-answer")
async def process_answer(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
):
    """
    Receive candidate audio answer:
      1. Transcribe with Faster-Whisper
      2. Store in memory
      3. Handle introduction stage OR interview flow
      4. Generate next question / follow-up
    """

    session = get_session(session_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please restart."
        )

    # Allow only valid interview states
    if session.interview_state not in ("in_progress", "waiting_intro"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot process answer in state '{session.interview_state}'",
        )

    tmp_path = None

    try:
        # ─────────────────────────────────────────────────────
        # Save uploaded audio
        # ─────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as tmp:

            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # ─────────────────────────────────────────────────────
        # Step 1 — Transcribe
        # ─────────────────────────────────────────────────────
        transcription = transcribe_audio(tmp_path)

        if not transcription or not is_meaningful_response(transcription):

            return {
                "success": False,
                "transcription": transcription or "",
                "message": "No clear response detected. Please speak louder and try again.",
                "next_action": "retry",
            }

        logger.info(
            f"Answer #{session.question_count}: '{transcription[:60]}'"
        )

        # ─────────────────────────────────────────────────────
        # Store user message
        # ─────────────────────────────────────────────────────
        session.add_message(
            role="user",
            content=transcription,
            metadata={
                "type": "answer",
                "question_number": session.question_count,
            },
        )

        # ─────────────────────────────────────────────────────
        # INTRODUCTION STAGE
        # ─────────────────────────────────────────────────────
        if session.interview_state == "waiting_intro":

            logger.info(
                "Processing candidate introduction only"
            )

            # Move into actual interview
            session.set_state("in_progress")

            # Generate FIRST technical question
            next_q = generate_question(
                resume_data=session.resume_data,
                conversation_history=session.get_conversation_history(),
                asked_questions=session.get_asked_questions(),
            )

            # Clean response for TTS
            next_q = next_q.replace('"', "").strip()

            # Limit question length
            if len(next_q) > 120:
                next_q = next_q[:120]

            session.add_message(
                role="assistant",
                content=next_q,
                is_question=True,
                metadata={
                    "type": "question",
                    "question_number": 1,
                },
            )

            return {
                "success": True,
                "transcription": transcription,
                "next_response": next_q,
                "next_action": "continue",
                "question_number": 1,
                "question_type": "new",
                "is_complete": False,
            }

        # ─────────────────────────────────────────────────────
        # NORMAL INTERVIEW FLOW
        # ─────────────────────────────────────────────────────

        last_question = (
            session.asked_questions[-1]
            if session.asked_questions
            else ""
        )

        # ─────────────────────────────────────────────────────
        # Interview complete?
        # ─────────────────────────────────────────────────────
        if is_interview_complete(session.question_count):

            closing = generate_closing_statement(
                candidate_name=session.resume_data.get(
                    "name",
                    "there"
                ),
                question_count=session.question_count,
            )

            session.add_message(
                role="assistant",
                content=closing,
                is_question=False,
            )

            session.set_state("closing")

            return {
                "success": True,
                "transcription": transcription,
                "next_response": closing,
                "next_action": "generate_feedback",
                "question_number": session.question_count,
                "is_complete": True,
            }

        # ─────────────────────────────────────────────────────
        # Follow-up OR New Question
        # ─────────────────────────────────────────────────────
        if should_ask_followup(
            last_question,
            transcription
        ):

            next_q = generate_followup_question(
                last_question,
                transcription,
                session.resume_data,
            )

            question_type = "followup"

        else:

            next_q = generate_question(
                resume_data=session.resume_data,
                conversation_history=session.get_conversation_history(),
                asked_questions=session.get_asked_questions(),
            )

            question_type = "new"

        # ─────────────────────────────────────────────────────
        # Clean question for TTS
        # ─────────────────────────────────────────────────────
        next_q = next_q.replace('"', "").strip()

        # Prevent huge TTS messages
        if len(next_q) > 120:
            next_q = next_q[:120]

        session.add_message(
            role="assistant",
            content=next_q,
            is_question=True,
            metadata={
                "type": question_type,
                "question_number": session.question_count + 1,
            },
        )

        return {
            "success": True,
            "transcription": transcription,
            "next_response": next_q,
            "next_action": "continue",
            "question_number": session.question_count,
            "question_type": question_type,
            "is_complete": False,
        }

    except Exception as e:

        logger.error(f"process-answer error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── Feedback ──────────────────────────────────────────────────────────────────

@app.post("/generate-feedback")
async def generate_feedback_endpoint(body: dict):
    """Generate the final comprehensive feedback report for the session."""
    session = get_session(body.get("session_id", ""))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        feedback = generate_feedback(
            resume_data=session.resume_data,
            conversation_history=session.get_conversation_history(),
            transcript=session.get_transcript(),
        )
        session.set_feedback(feedback)
        saved = session.save_to_disk()

        return {
            "success":          True,
            "feedback":         feedback,
            "session_id":       session.session_id,
            "transcript_saved": saved is not None,
            "tts_summary":      format_feedback_for_tts(feedback),
        }

    except Exception as e:
        logger.error(f"Feedback generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── TTS ───────────────────────────────────────────────────────────────────────

@app.post("/synthesize-speech")
async def synthesize_speech(body: dict):
    """
    Convert text to speech using pyttsx3 (offline).
    Returns raw WAV bytes for browser playback via Streamlit.
    """
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        wav_bytes = get_tts_audio_bytes(text)
        if not wav_bytes:
            raise HTTPException(status_code=500, detail="TTS produced empty audio")

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"},
        )

    except Exception as e:
        logger.error(f"TTS endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session ───────────────────────────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session_status(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    session = get_session(session_id)
    if session:
        session.save_to_disk()
        delete_session(session_id)
    return {"success": True, "message": "Session ended and saved"}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    model = os.getenv("OLLAMA_MODEL", "llama3")

    print(f"\n🎙️  Voice Interview Agent (Offline Edition)")
    print(f"   LLM:     Ollama → {model}")
    print(f"   STT:     Faster-Whisper (local)")
    print(f"   TTS:     pyttsx3 (offline)")
    print(f"   API:     http://{host}:{port}")
    print(f"   Docs:    http://{host}:{port}/docs\n")

    uvicorn.run("app:app", host=host, port=port, reload=True, log_level="info")
