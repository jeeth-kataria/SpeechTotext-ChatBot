# backend/main.py

import os
import shutil
import uuid
import time
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

WHISPER_MODEL_SIZE = "small"
WHISPER_FP16 = False

ACTUAL_GEMINI_MODEL_TO_USE = "gemini-1.5-flash-latest"

# --- SIMPLIFIED SYSTEM PROMPT V2 ---
GEMINI_SYSTEM_INSTRUCTION = """
ROLE: Simulate an AI assistant providing general medical information for a hackathon demo.
LANGUAGE: Respond ONLY in the same language as the user's last message. Default to English if unsure.
SCOPE: Discuss common symptoms and general home care for MILD conditions. Provide general medical information.
RESTRICTIONS:
- NEVER give a diagnosis. State "Only a doctor can diagnose."
- NEVER suggest specific prescription or OTC drug names/dosages. Use general categories only (e.g., 'pain relievers').
- REFUSE non-medical questions politely (e.g., "My focus is health information only.").
- ALWAYS recommend consulting a qualified doctor for diagnosis, treatment decisions, serious symptoms, or persistent issues. Redirect to emergency services for urgent situations (e.g., severe chest pain).
OUTPUT FORMAT: Provide only the helpful information requested within the scope. Do NOT add any disclaimers, warnings, or text about being an AI at the end. Conclude naturally.
"""
# --- End Simplified System Prompt ---

TEMP_AUDIO_DIR = "temp_audio_files"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

app = FastAPI(
    title="Medical Advisor Chatbot Backend",
    description="Handles STT via Whisper and LLM chat via Gemini (Simplified Prompt).",
    version="1.3.0" # Incremented version
)

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:3005", # Your frontend port
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Models ---
whisper_model = None
gemini_llm = None
active_gemini_chat_session = None

# --- Server Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Loads models when the FastAPI application starts."""
    global whisper_model, gemini_llm, ACTUAL_GEMINI_MODEL_TO_USE, GOOGLE_API_KEY, active_gemini_chat_session

    print("--- Backend Startup Sequence Initiated ---")

    # Load Whisper Model
    print(f"Loading Whisper model: '{WHISPER_MODEL_SIZE}' (FP16: {WHISPER_FP16})...")
    if not os.path.exists(os.path.expanduser(f"~/.cache/whisper/{WHISPER_MODEL_SIZE}.pt")):
        print(f"Model file for '{WHISPER_MODEL_SIZE}' not found in cache. Download may be required.")
    try:
        start_time = time.time()
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        end_time = time.time()
        print(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading Whisper model: {e} !!!")
        print(traceback.format_exc())
        whisper_model = None

    # Configure and Initialize Gemini Model Object
    if GOOGLE_API_KEY:
        print(f"Attempting to initialize Gemini model: {ACTUAL_GEMINI_MODEL_TO_USE} with SIMPLIFIED system instruction.")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Initialize with the SIMPLIFIED system instruction
            gemini_llm = genai.GenerativeModel(
                ACTUAL_GEMINI_MODEL_TO_USE,
                system_instruction=GEMINI_SYSTEM_INSTRUCTION # Use simplified prompt
            )
            print(f"Gemini model object ({ACTUAL_GEMINI_MODEL_TO_USE}) created successfully.")
            initialize_chat_session_internal() # Initialize chat session
        except Exception as e:
            print(f"!!! CRITICAL ERROR initializing Gemini model object: {e} !!!")
            print(traceback.format_exc())
            gemini_llm = None
    else:
        print("WARNING: GOOGLE_API_KEY not found. Gemini LLM will be unavailable.")
        gemini_llm = None
    
    print("--- Backend Startup Sequence Complete ---")


def initialize_chat_session_internal():
    """Internal function to create/reset the chat session using the configured gemini_llm."""
    global active_gemini_chat_session, gemini_llm
    if gemini_llm is None:
        print("DEBUG initialize_chat_session_internal: gemini_llm is None, cannot create session.")
        active_gemini_chat_session = None
        return False

    print("DEBUG initialize_chat_session_internal: Creating new chat session from configured LLM object.")
    try:
        # History starts empty; system prompt is part of the model object now
        active_gemini_chat_session = gemini_llm.start_chat(history=[])
        print("DEBUG initialize_chat_session_internal: New chat session created successfully.")
        return True
    except Exception as e:
        print(f"!!! ERROR creating new chat session: {e} !!!")
        print(traceback.format_exc())
        active_gemini_chat_session = None
        return False

# --- API Endpoints ---

@app.get("/ping", tags=["Health"])
async def ping():
    """Simple health check endpoint."""
    print("--- /ping endpoint hit ---")
    return {"message": "pong from medical chatbot backend v1.3", "whisper_loaded": whisper_model is not None, "gemini_loaded": gemini_llm is not None}

@app.post("/transcribe", tags=["Transcription"])
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """Receives audio file, transcribes using Whisper with auto language detection."""
    print(f"--- /transcribe endpoint hit. Received file: {audio_file.filename} (type: {audio_file.content_type}) ---")
    if whisper_model is None:
        print("!!! ERROR /transcribe: Whisper model not loaded. !!!")
        raise HTTPException(status_code=503, detail="Whisper model not available on server.")
    file_path = None
    try:
        allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4"}
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        if not file_extension or file_extension not in allowed_extensions:
             print(f"WARNING: Received file with potentially unsupported extension '{file_extension}'. Defaulting to .webm for saving.")
             file_extension = ".webm"
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(TEMP_AUDIO_DIR, temp_filename)
        
        print(f"Saving uploaded audio to: {file_path}")
        t_start_save = time.time()
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        t_end_save = time.time()
        print(f"Audio saved. File size: {os.path.getsize(file_path)} bytes. Save duration: {t_end_save - t_start_save:.2f}s")
        
        print(f"Attempting Whisper transcription (Model: {WHISPER_MODEL_SIZE}, Language: Auto-Detect)")
        t_start_transcribe = time.time()

        result = whisper_model.transcribe(
            file_path, 
            fp16=WHISPER_FP16
        )
        
        t_end_transcribe = time.time()
        transcription_duration = t_end_transcribe - t_start_transcribe
        print(f"Transcription duration: {transcription_duration:.2f} seconds")

        transcribed_text = result["text"]
        detected_language = result.get("language", "unknown")
        print(f"Whisper Transcribed (Auto-Detected): '{transcribed_text}' (Detected lang: {detected_language})")

        return {"transcription": transcribed_text, "language": detected_language, "duration_sec": transcription_duration}
        
    except Exception as e:
        print(f"!!! ERROR during /transcribe processing: {e} !!!")
        print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Server error during transcription: {str(e)}")
        
    finally:
        if file_path and os.path.exists(file_path):
            print(f"Cleaning up temporary audio file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e_rem:
                print(f"Warning: Could not remove temp file {file_path}: {e_rem}")


@app.post("/chat", tags=["Chat"])
async def chat_with_llm_endpoint(payload: dict):
    """Receives text, gets response from Gemini model (using simplified system prompt)."""
    user_text = payload.get("text")
    print(f"--- /chat endpoint hit. Received text: '{user_text}' ---")

    if not user_text:
        print("!!! ERROR /chat: No text provided in payload. !!!")
        raise HTTPException(status_code=400, detail="No text provided in chat payload.")

    if gemini_llm is None:
        print("DEBUG /chat: gemini_llm is None. Responding with llm_offline.")
        return {"llm_response": None, "status": "llm_offline", "detail": "LLM (Gemini) is not available on the server."}

    current_chat_session = active_gemini_chat_session
    if current_chat_session is None:
        print("DEBUG /chat: active_gemini_chat_session is None. Attempting to re-initialize.")
        if not initialize_chat_session_internal():
             print("!!! ERROR /chat: Failed to re-initialize chat session. !!!")
             return {"llm_response": None, "status": "llm_error", "detail": "LLM chat session could not be re-initialized on backend."}
        current_chat_session = active_gemini_chat_session

    try:
        print(f"DEBUG /chat: Sending to Gemini model '{ACTUAL_GEMINI_MODEL_TO_USE}' (simplified prompt): '{user_text}'")
        t_start_llm = time.time()
        
        response = current_chat_session.send_message(user_text) 
        
        t_end_llm = time.time()
        print(f"DEBUG /chat: Received response object from Gemini in {t_end_llm - t_start_llm:.2f}s.")

        llm_response_text = None
        response_status = "success"

        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name
            llm_response_text = f"UI_MSG: Blocked - {reason}"
            response_status = "llm_blocked"
            print(f"DEBUG /chat: Gemini response blocked: {reason}")
        elif not response.parts:
             llm_response_text = "UI_MSG: Response filtered or empty by Gemini."
             response_status = "llm_filtered"
             print("DEBUG /chat: Gemini response parts are empty.")
        else:
            llm_response_text = response.text.strip()
            print(f"DEBUG /chat: Gemini successfully responded (length: {len(llm_response_text)}).")
            # No manual disclaimer check/append needed based on the simplified prompt

        return {"llm_response": llm_response_text, "status": response_status}

    except Exception as e:
        print(f"!!! EXCEPTION in /chat during Gemini send_message: {type(e).__name__} - {str(e)} !!!")
        print(traceback.format_exc())
        return {"llm_response": None, "status": "llm_error", "detail": f"Backend LLM Error: {type(e).__name__}"}


@app.post("/clear_chat_session", tags=["Chat"])
async def clear_chat_session_endpoint():
    """Clears the current backend chat session history by re-initializing it."""
    global active_gemini_chat_session
    if initialize_chat_session_internal():
        print("--- /clear_chat_session endpoint hit. Backend Gemini chat session reset. ---")
        return {"message": "Chat session cleared/reset on backend"}
    else:
        print("--- /clear_chat_session endpoint hit but failed to reset session (LLM might be down). ---")
        return {"message": "Chat session reset attempted on backend (check backend logs)."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    port_to_use = 8001 # Use the port your frontend expects
    print(f"--- Starting FastAPI backend server using Uvicorn ---")
    print(f"Host: 0.0.0.0, Port: {port_to_use}")
    print(f"API docs will be available at http://localhost:{port_to_use}/docs")
    print("Models loading during startup...")
    uvicorn.run("main:app", host="0.0.0.0", port=port_to_use, reload=True, log_level="info")