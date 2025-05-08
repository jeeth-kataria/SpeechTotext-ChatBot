import os
import shutil
import uuid
import traceback # For detailed exception logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # !!! REPLACE OR ENSURE ENV VAR IS SET !!!
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    print("FATAL: GOOGLE_API_KEY is not set or is placeholder. Backend will not function correctly with LLM.")
    # For testing, we'll let it proceed but LLM calls will fail. In production, you might exit.

WHISPER_MODEL_SIZE = "base"
ACTUAL_GEMINI_MODEL_TO_USE = "gemini-1.5-flash-latest" # Default, can be overridden
TEMP_AUDIO_DIR = "temp_audio_files"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost:3005"] # Add other frontend origins if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
gemini_llm = None
# We will manage chat session per request for more statelessness in this debug version,
# or re-initialize it if it becomes None.
# A global session can cause issues with stale states if not managed carefully.
# For a simple demo, a global one that gets reset is okay.
active_gemini_chat_session = None


@app.on_event("startup")
async def startup_event():
    global whisper_model, gemini_llm, ACTUAL_GEMINI_MODEL_TO_USE, GOOGLE_API_KEY
    print("--- Backend Startup ---")
    print("Loading Whisper model...")
    try:
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully.")
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading Whisper model: {e} !!!")
        print(traceback.format_exc())
        whisper_model = None # Ensure it's None if failed
    
    if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
        print(f"Attempting to initialize Gemini model: {ACTUAL_GEMINI_MODEL_TO_USE} with provided API key.")
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            system_instruction_content = "You are a helpful AI assistant."
            # Initialize the main LLM model object
            gemini_llm = genai.GenerativeModel(
                ACTUAL_GEMINI_MODEL_TO_USE,
                system_instruction=system_instruction_content if "1.5" in ACTUAL_GEMINI_MODEL_TO_USE else None
            )
            print(f"Gemini model object ({ACTUAL_GEMINI_MODEL_TO_USE}) created successfully.")
            # We will initialize the chat session itself on first use or when cleared.
        except Exception as e:
            print(f"!!! CRITICAL ERROR initializing Gemini model object: {e} !!!")
            print(traceback.format_exc())
            gemini_llm = None
    else:
        print("WARNING: GOOGLE_API_KEY not set or is placeholder. Gemini LLM will be unavailable.")
        gemini_llm = None
    print("--- Backend Startup Complete ---")


def get_or_create_chat_session():
    global active_gemini_chat_session, gemini_llm, ACTUAL_GEMINI_MODEL_TO_USE
    if gemini_llm is None:
        print("DEBUG get_or_create_chat_session: gemini_llm is None, cannot create session.")
        return None
    
    if active_gemini_chat_session is None:
        print("DEBUG get_or_create_chat_session: No active session, creating new one.")
        try:
            # For models not supporting system_instruction directly in GenerativeModel init
            initial_history = []
            if "1.5" not in ACTUAL_GEMINI_MODEL_TO_USE: # Example for older models
                 system_instruction_content = "You are a helpful AI assistant."
                 initial_history = [
                    {"role": "user", "parts": [{"text": system_instruction_content}]},
                    {"role": "model", "parts": [{"text": "Okay, I will be helpful."}]}
                 ]
            active_gemini_chat_session = gemini_llm.start_chat(history=initial_history)
            print("DEBUG get_or_create_chat_session: New chat session created.")
        except Exception as e:
            print(f"!!! ERROR creating new chat session: {e} !!!")
            print(traceback.format_exc())
            active_gemini_chat_session = None # Ensure it's None on failure
    return active_gemini_chat_session


@app.post("/transcribe")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    print(f"--- /transcribe endpoint hit. Received file: {audio_file.filename} (type: {audio_file.content_type}) ---")
    if whisper_model is None:
        print("!!! ERROR /transcribe: Whisper model not loaded. !!!")
        raise HTTPException(status_code=503, detail="Whisper model not available on server.")

    file_path = None
    try:
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        if not file_extension: file_extension = ".webm" # Common browser default
        
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(TEMP_AUDIO_DIR, temp_filename)
        
        print(f"Saving uploaded audio to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"Audio saved. File size: {os.path.getsize(file_path)} bytes.")
        
        print(f"Attempting Whisper transcription for: {file_path}")
        result = whisper_model.transcribe(file_path, fp16=False)
        transcribed_text = result["text"]
        print(f"Whisper Transcribed Successfully: '{transcribed_text}'")
        return {"transcription": transcribed_text}
    except Exception as e:
        print(f"!!! ERROR during /transcribe processing: {e} !!!")
        print(traceback.format_exc()) # Print full traceback
        raise HTTPException(status_code=500, detail=f"Server error during transcription: {str(e)}")
    finally:
        if file_path and os.path.exists(file_path):
            print(f"Cleaning up temporary audio file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e_rem:
                print(f"Warning: Could not remove temp file {file_path}: {e_rem}")


@app.post("/chat")
async def chat_with_llm_endpoint(payload: dict):
    user_text = payload.get("text")
    print(f"--- /chat endpoint hit. Received text: '{user_text}' ---")

    if not user_text:
        print("!!! ERROR /chat: No text provided in payload. !!!")
        raise HTTPException(status_code=400, detail="No text provided in chat payload.")

    if gemini_llm is None:
        print("DEBUG /chat: gemini_llm is None. Responding with llm_offline.")
        return {"llm_response": None, "status": "llm_offline", "detail": "LLM (Gemini) is not available on the server."}

    current_chat_session = get_or_create_chat_session()
    if current_chat_session is None:
        print("!!! ERROR /chat: Failed to get or create chat session. !!!")
        return {"llm_response": None, "status": "llm_error", "detail": "LLM chat session could not be initialized on backend."}

    try:
        print(f"DEBUG /chat: Sending to Gemini model '{ACTUAL_GEMINI_MODEL_TO_USE}': '{user_text}'")
        # The chat session manages its own history internally
        response = current_chat_session.send_message(user_text) 
        print("DEBUG /chat: Received response object from Gemini.")

        llm_response_text = None
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason.name
            llm_response_text = f"UI_MSG: Blocked - {reason}"
            print(f"DEBUG /chat: Gemini response blocked: {reason}")
        elif not response.parts:
            llm_response_text = "UI_MSG: Response filtered or empty by Gemini."
            print("DEBUG /chat: Gemini response parts are empty.")
        else:
            llm_response_text = response.text.strip()
            print(f"DEBUG /chat: Gemini successfully responded: '{llm_response_text}'")
        
        return {"llm_response": llm_response_text, "status": "success"}
    except Exception as e:
        print(f"!!! EXCEPTION in /chat during Gemini send_message: {type(e).__name__} - {str(e)} !!!")
        print(traceback.format_exc()) # Print full traceback
        return {"llm_response": None, "status": "llm_error", "detail": f"Backend LLM Error: {type(e).__name__}"}


@app.post("/clear_chat_session")
async def clear_chat_session_endpoint():
    global active_gemini_chat_session
    active_gemini_chat_session = None # This will cause get_or_create_chat_session to make a new one
    print("--- /clear_chat_session endpoint hit. Backend Gemini chat session marked for reset. ---")
    return {"message": "Chat session cleared on backend"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for backend...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)