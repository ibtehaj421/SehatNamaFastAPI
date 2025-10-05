

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from typing import Literal
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import base64
import requests
import json
from supabase import create_client, Client
# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Urdu STT API",
    description="Urdu Speech-to-Text using Groq Whisper",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sehat-nama.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)

# Initialize UpliftAI configuration
UPLIFTAI_API_KEY = os.getenv("UPLIFTAI_API_KEY")
UPLIFTAI_BASE_URL = "https://api.upliftai.org/v1"

# Initialize Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully")
else:
    supabase = None
    print("Supabase configuration not found - medical history storage disabled")

print(f"UpliftAI API Key: {'Set' if UPLIFTAI_API_KEY else 'Not Set'}")
# Supported audio formats
SUPPORTED_FORMATS = {'flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg','opus', 'wav', 'webm'}


@app.get("/")
async def root():
    """Health check"""
    return {"status": "running", "message": "Urdu STT API"}


@app.post("/transcribe")
async def transcribe_urdu_audio(
    file: UploadFile = File(..., description="Audio file in Urdu"),
    model: Literal["whisper-large-v3-turbo", "whisper-large-v3"] = Form(
        default="whisper-large-v3-turbo",
        description="Whisper model (turbo is faster)"
    )
):
    
    print(f"Transcribing file: {file.filename}, size: {file.size}, model: {model}")
    # Validate file format
    file_ext = Path(file.filename).suffix.lower().lstrip('.')
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported format. Use: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe with Groq Whisper (Urdu language)
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                language="ur",  # Urdu language code
                response_format="verbose_json",
                temperature=0.0
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Return response
        return {
            "text": transcription.text,
            "language": transcription.language,
            "duration": transcription.duration,
            "segments": transcription.segments
        }
    
    except Exception as e:
        # Clean up on error
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(..., description="Text to convert to speech"),
    voice_id: str = Form(
        default="v_meklc281",
        description="UpliftAI voice ID"
    ),
    output_format: str = Form(
        default="MP3_22050_32",
        description="Audio output format"
    ),
    save_file: bool = Form(
        default=False,
        description="Save to file (testing) or stream (production)"
    )
):
    """
    Convert text to speech using UpliftAI
    
    - **text**: Text to convert to speech
    - **voice_id**: Voice ID from UpliftAI (default: v_meklc281)
    - **output_format**: Audio format (MP3_22050_32, MP3_44100_128, etc.)
    - **save_file**: True = save file, False = stream audio (production)
    """
    
    if not UPLIFTAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="UPLIFTAI_API_KEY not configured in .env file"
        )
    
    try:
        # Prepare UpliftAI TTS request
        url = f"{UPLIFTAI_BASE_URL}/synthesis/text-to-speech"
        headers = {
            "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voiceId": voice_id,
            "outputFormat": output_format
        }
        
        # Call UpliftAI TTS API
        response = requests.post(url, json=payload, headers=headers)
        
        # if response.status_code != 200:
        #     raise HTTPException(
        #         status_code=response.status_code,
        #         detail=f"UpliftAI TTS failed: {response.text}"
        #     )
        
        # # Get audio data
        # result = response.json()
        
        # # Audio is typically base64 encoded or URL
        # if "audioContent" in result:
        #     # Base64 encoded audio
        #     audio_data = base64.b64decode(result["audioContent"])
        # elif "url" in result:
        #     # Download from URL
        #     audio_response = requests.get(result["url"])
        #     audio_data = audio_response.content
        # else:
        #     raise HTTPException(
        #         status_code=500,
        #         detail="Unexpected response format from UpliftAI"
        #     )
        content_type = response.headers.get("Content-Type", "")

        if "application/json" in content_type:
            result = response.json()

            if "audioContent" in result:
                # Base64 encoded audio
                audio_data = base64.b64decode(result["audioContent"])
            elif "url" in result:
                # Download from URL
                audio_response = requests.get(result["url"])
                audio_data = audio_response.content
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected JSON response format from UpliftAI"
                )

        elif "audio" in content_type:
            # Raw audio returned directly
            audio_data = response.content

        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected response from UpliftAI: {response.text}"
            )
        
        # Save file mode (for testing)
        if save_file:
            filename = f"tts_output_{os.urandom(4).hex()}.mp3"
            filepath = Path("audio_outputs") / filename
            filepath.parent.mkdir(exist_ok=True)
            
            with open(filepath, "wb") as f:
                f.write(audio_data)
            
            return {
                "message": "Audio saved successfully",
                "filepath": str(filepath),
                "filename": filename,
                "size_bytes": len(audio_data)
            }
        
        # Streaming mode (for production)
        else:
            return StreamingResponse(
                iter([audio_data]),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text-to-speech failed: {str(e)}"
        )


# ========== LLM SESSION ENDPOINTS ==========
try:
    from llm import UrduMedicalHistorySystem
    from llm import medical_system as _unused  # if llm creates app instance, ignore
except Exception:
    # Import lazily if running as script
    from importlib import import_module
    llm_mod = import_module('llm')
    UrduMedicalHistorySystem = getattr(llm_mod, 'UrduMedicalHistorySystem')

# Single shared LLM system instance
llm_system = UrduMedicalHistorySystem()
llm_sessions = {}


@app.post('/api/start-interview')
async def api_start_interview():
    try:
        result = llm_system.start_interview()
        import uuid
        session_id = str(uuid.uuid4())
        llm_sessions[session_id] = result['state']

        # Extract clean message
        ai_message = result['ai_message']
        if hasattr(ai_message, "choices"):
            ai_message = ai_message.choices[0].message.content

        return {'session_id': session_id, 'message': ai_message}
    except Exception as e:
        return {'error': 'start-interview failed', 'details': str(e)}


@app.post('/api/start-interview-with-voice')
async def api_start_interview_with_voice():
    """Start interview and return both text + audio for the first question"""
    try:
        # Start the interview to get the first question
        result = llm_system.start_interview()
        import uuid
        session_id = str(uuid.uuid4())
        llm_sessions[session_id] = result['state']

        # Extract clean message
        ai_message = result['ai_message']
        if hasattr(ai_message, "choices"):
            ai_message = ai_message.choices[0].message.content

        # Convert the first question to speech using UpliftAI
        if not UPLIFTAI_API_KEY:
            return {
                'session_id': session_id, 
                'message': ai_message,
                'audio_base64': None,
                'tts_error': 'TTS not configured'
            }

        try:
            # Call UpliftAI TTS for the first question
            url = f"{UPLIFTAI_BASE_URL}/synthesis/text-to-speech"
            headers = {
                "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "text": ai_message,
                "voiceId": "v_meklc281",  # Default Urdu voice
                "outputFormat": "MP3_22050_32"
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            # Handle the audio response
            content_type = response.headers.get("Content-Type", "")
            
            if "application/json" in content_type:
                result_tts = response.json()
                if "audioContent" in result_tts:
                    audio_data = base64.b64decode(result_tts["audioContent"])
                elif "url" in result_tts:
                    audio_response = requests.get(result_tts["url"])
                    audio_data = audio_response.content
                else:
                    raise Exception("Unexpected JSON response from TTS")
            elif "audio" in content_type:
                audio_data = response.content
            else:
                raise Exception(f"Unexpected response: {response.text}")
            
            # Return base64 encoded audio for frontend
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {
                'session_id': session_id,
                'message': ai_message,
                'audio_base64': audio_base64,
                'audio_format': 'mp3'
            }
            
        except Exception as tts_error:
            # Return text even if TTS fails
            return {
                'session_id': session_id,
                'message': ai_message,
                'audio_base64': None,
                'tts_error': str(tts_error)
            }
        
    except Exception as e:
        return {'error': 'start-interview failed', 'details': str(e)}


from pydantic import BaseModel

class SendMessageRequest(BaseModel):
    session_id: str
    message: str

class StoreMedicalHistoryRequest(BaseModel):
    user_email: str
    medical_data: dict

@app.post('/api/send-message')
async def api_send_message(req: SendMessageRequest):
    try:
        if req.session_id not in llm_sessions:
            return {'error': 'session not found'}

        state = llm_sessions[req.session_id]
        result = llm_system.process_user_message(state, req.message)
        llm_sessions[req.session_id] = result['state']
        print('Send Message')      
        print(result)
        ai_message = result['ai_message']
        if hasattr(ai_message, "choices") and ai_message.choices:
            print('im here 1')
            ai_message = ai_message.choices[0].message.content
        elif hasattr(ai_message, "content"):
            print('im here 2')
            ai_message = ai_message.content
        else:
            print('im here 3')
            ai_message = str(ai_message)
        return {
            'message': ai_message,
            'collected_data': result['collected_data'],
            'is_complete': result['is_complete']
        }
    except Exception as e:
        return {'error': 'send-message failed', 'details': str(e)}


@app.post('/api/send-message-with-voice')
async def api_send_message_with_voice(req: SendMessageRequest):
    """Send message and return both text + audio response"""
    try:
        if req.session_id not in llm_sessions:
            return {'error': 'session not found'}

        state = llm_sessions[req.session_id]
        print(f"State before processing: {state.get('current_section')}")
        result = llm_system.process_user_message(state, req.message)
        llm_sessions[req.session_id] = result['state']
        print(f"Result keys: {result.keys()}")
        print(f"AI message raw: {result.get('ai_message')}")
        print(f"AI message type: {type(result.get('ai_message'))}")
        # Extract AI message
        ai_message = result['ai_message']
        if hasattr(ai_message, "choices") and ai_message.choices:
            ai_message = ai_message.choices[0].message.content
        elif hasattr(ai_message, "content"):
            ai_message = ai_message.content
        else:
            ai_message = str(ai_message)

        # Convert response to speech
        audio_base64 = None
        tts_error = None
        
        if UPLIFTAI_API_KEY and ai_message:
            try:
                url = f"{UPLIFTAI_BASE_URL}/synthesis/text-to-speech"
                headers = {
                    "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "text": ai_message,
                    "voiceId": "v_meklc281",
                    "outputFormat": "MP3_22050_32"
                }
                
                response = requests.post(url, json=payload, headers=headers)
                content_type = response.headers.get("Content-Type", "")
                
                if "application/json" in content_type:
                    result_tts = response.json()
                    if "audioContent" in result_tts:
                        audio_data = base64.b64decode(result_tts["audioContent"])
                    elif "url" in result_tts:
                        audio_response = requests.get(result_tts["url"])
                        audio_data = audio_response.content
                    else:
                        raise Exception("Unexpected JSON response")
                elif "audio" in content_type:
                    audio_data = response.content
                else:
                    raise Exception(f"Unexpected response: {response.text}")
                
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
            except Exception as e:
                tts_error = str(e)

        return {
            'message': ai_message,
            'collected_data': result['collected_data'],
            'is_complete': result['is_complete'],
            'audio_base64': audio_base64,
            'audio_format': 'mp3',
            'tts_error': tts_error
        }
        
    except Exception as e:
        return {'error': 'send-message failed', 'details': str(e)}


@app.post('/api/text-to-audio')
async def api_text_to_audio(
    text: str = Form(...),
    voice_id: str = Form(default="v_meklc281")
):
    """Convert text to audio and return base64 encoded"""
    if not UPLIFTAI_API_KEY:
        raise HTTPException(status_code=500, detail="TTS not configured")
    
    try:
        url = f"{UPLIFTAI_BASE_URL}/synthesis/text-to-speech"
        headers = {
            "Authorization": f"Bearer {UPLIFTAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voiceId": voice_id,
            "outputFormat": "MP3_22050_32"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        content_type = response.headers.get("Content-Type", "")
        
        if "application/json" in content_type:
            result = response.json()
            if "audioContent" in result:
                audio_data = base64.b64decode(result["audioContent"])
            elif "url" in result:
                audio_response = requests.get(result["url"])
                audio_data = audio_response.content
            else:
                raise HTTPException(status_code=500, detail="Unexpected response format")
        elif "audio" in content_type:
            audio_data = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response: {response.text}")
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            'audio_base64': audio_base64,
            'audio_format': 'mp3',
            'text': text
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.get('/api/get-history')
async def api_get_history(session_id: str, view: str = 'patient'):
    """Return formatted history for a session. view='patient'|'doctor'"""
    if session_id not in llm_sessions:
        return { 'error': 'session not found' }

    state = llm_sessions[session_id]

    try:
        history = llm_system.get_history_view(state, view=view)
    except Exception as e:
        return { 'error': 'failed to build history', 'details': str(e) }

    return { 'session_id': session_id, 'view': view, 'history': history }


@app.websocket("/ws/interview/{session_id}")
async def websocket_interview(websocket: WebSocket, session_id: str):
    """WebSocket for streaming responses (exposes same behavior as llm.py websocket)
    """
    await websocket.accept()

    # Initialize or get session
    if session_id not in llm_sessions:
        result = llm_system.start_interview()
        llm_sessions[session_id] = result['state']
        await websocket.send_json({
            "type": "message",
            "content": result['ai_message']
        })

    state = llm_sessions[session_id]

    while True:
        try:
            data = await websocket.receive_json()
        except Exception:
            break
        user_message = data.get('message')

        # Stream response tokens
        collected_text = ""
        try:
            async for chunk in llm_system.process_user_message_streaming(state, user_message):
                if isinstance(chunk, dict) and 'content' in chunk:
                    token = chunk['content']
                    collected_text += token
                    await websocket.send_json({"type": "token", "content": token})
        except Exception as e:
            # send an error and continue
            await websocket.send_json({"type": "error", "message": str(e)})

        # Send completion
        await websocket.send_json({
            "type": "complete",
            "collected_data": state.get('collected_data', {}),
            "is_complete": state.get('all_sections_done', False)
        })

        # Update session
        llm_sessions[session_id] = state
@app.post('/api/store-medical-history')
async def api_store_medical_history(req: StoreMedicalHistoryRequest):
    """
    Store medical interview results in Supabase after translation
    Takes Urdu medical data, translates to English, and stores both versions
    """
    if not supabase:
        raise HTTPException(
            status_code=500, 
            detail="Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables."
        )
    
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured for translation"
        )
    
    try:
        # Prepare the original Urdu data
        urdu_version = req.medical_data
        
        # Translate each field to English using Groq
        english_version = {}
        
        # Iterate through each section and field in the medical data
        for section, section_data in urdu_version.items():
            if isinstance(section_data, dict):
                english_version[section] = {}
                for field, urdu_value in section_data.items():
                    if isinstance(urdu_value, str) and urdu_value.strip():
                        # Translate this Urdu text to English
                        english_value = await translate_urdu_to_english(urdu_value)
                        english_version[section][field] = english_value
                    else:
                        # Non-string or empty values remain as-is
                        english_version[section][field] = urdu_value
            else:
                # Non-dict values remain as-is
                english_version[section] = section_data
        
        # Store both versions in Supabase
        result = supabase.table('medical_history').insert({
            'email': req.user_email,
            'urdu_version': urdu_version,
            'english_version': english_version
        }).execute()
        
        if result.data:
            return {
                'success': True,
                'message': 'Medical history stored successfully',
                'record_id': result.data[0]['id'],
                'urdu_version': urdu_version,
                'english_version': english_version
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to store medical history in database"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store medical history: {str(e)}"
        )


async def translate_urdu_to_english(urdu_text: str) -> str:
    """
    Translate Urdu text to English using Groq LLM
    """
    try:
        translation_prompt = f"""
Translate the following Urdu text to English. Provide only the direct translation without any additional commentary or explanation.

Urdu Text: {urdu_text}

English Translation:"""

        # Use Groq for translation
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": translation_prompt
                }
            ],
            model="llama-3.1-8b-instant",  # Fast model for translation
            temperature=0.1,  # Low temperature for consistent translation
            max_tokens=512
        )
        
        english_translation = chat_completion.choices[0].message.content.strip()
        
        # Clean up the response (remove any prefix like "English Translation:" if present)
        if english_translation.startswith("English Translation:"):
            english_translation = english_translation.replace("English Translation:", "").strip()
        
        return english_translation
        
    except Exception as e:
        # If translation fails, return the original text with error note
        return f"[Translation Error: {str(e)}] {urdu_text}"


@app.get('/api/example-store-usage')
async def api_example_store_usage():
    """
    Example showing how to use the store-medical-history endpoint
    This shows the expected format of medical_data when interview is complete
    """
    example_medical_data = {
        "demographics": {
            "name": "احمد علی",
            "age": "پچیس سال",
            "gender": "مرد",
            "occupation": "انجینئر",
            "address": "کراچی، پاکستان",
            "contact": "03001234567"
        },
        "complaint": {
            "chief_complaint": "پیٹ میں درد ہو رہا ہے دو دنوں سے"
        },
        "hpc": {
            "pain_description": "درد تیز ہے اور کھانے کے بعد بڑھ جاتا ہے"
        },
        "systems": {
            "cardiovascular": "کوئی مسئلہ نہیں",
            "respiratory": "سانس لینے میں کوئی دشواری نہیں"
        },
        "pmh": {
            "previous_illnesses": "پہلے کبھی کوئی بیماری نہیں ہوئی"
        },
        "drugs": {
            "current_medications": "کوئی دوا نہیں لے رہا"
        },
        "social": {
            "smoking": "سگریٹ نہیں پیتا",
            "alcohol": "شراب نہیں پیتا"
        }
    }
    
    return {
        "message": "Example usage of store-medical-history endpoint",
        "endpoint": "/api/store-medical-history",
        "method": "POST",
        "request_body": {
            "user_email": "patient@example.com",
            "medical_data": example_medical_data
        },
        "note": "Call this endpoint when interview is complete (is_complete=true) with the collected_data from the session"
    }

@app.get('/api/get-all-histories')
async def api_get_all_histories(email: str):
    """
    Get all medical histories for a specific email address
    Returns a list of all medical history records with metadata
    
    Query Parameters:
    - email: The email address to fetch histories for
    """
    if not supabase:
        raise HTTPException(
            status_code=500, 
            detail="Supabase not configured. Please set SUPABASE_URL and SUPABASE_ANON_KEY environment variables."
        )
    
    try:
        # Get all medical history records for this email
        print(email)
        result = supabase.table('medical_history').select("*").eq('email', email).order('created_at', desc=True).execute()
        print(result)
        histories = []
        for record in result.data:
            # Create a summary of each record
            urdu_data = record['urdu_version']
            english_data = record['english_version']
            
            # Calculate completion stats
            total_sections = len(urdu_data) if isinstance(urdu_data, dict) else 0
            total_fields = 0
            if isinstance(urdu_data, dict):
                for section_data in urdu_data.values():
                    if isinstance(section_data, dict):
                        total_fields += len([v for v in section_data.values() if v and str(v).strip()])
            
            # Get chief complaint for preview
            chief_complaint = ""
            if isinstance(urdu_data, dict):
                for section_key, section_data in urdu_data.items():
                    if isinstance(section_data, dict):
                        for field_key, field_value in section_data.items():
                            if 'complaint' in field_key.lower() or 'chief' in field_key.lower():
                                chief_complaint = str(field_value)[:100] + "..." if len(str(field_value)) > 100 else str(field_value)
                                break
                    if chief_complaint:
                        break
            
            # Determine primary language based on content
            primary_language = "urdu"
            if isinstance(urdu_data, dict):
                sample_text = ""
                for section_data in urdu_data.values():
                    if isinstance(section_data, dict):
                        for value in section_data.values():
                            if isinstance(value, str) and value.strip():
                                sample_text = value
                                break
                        if sample_text:
                            break
                
                # Simple heuristic: if contains English alphabet more than Urdu, consider it English
                english_chars = sum(1 for c in sample_text if c.isalpha() and ord(c) < 128)
                total_chars = len([c for c in sample_text if c.isalpha()])
                if total_chars > 0 and english_chars / total_chars > 0.7:
                    primary_language = "english"
            
            histories.append({
                'id': record['id'],
                'created_at': record['created_at'],
                'primary_language': primary_language,
                'chief_complaint_preview': chief_complaint,
                'stats': {
                    'total_sections': total_sections,
                    'total_fields': total_fields,
                    'completion_status': 'Complete' if total_sections >= 5 else 'Partial'
                },
                'urdu_version': urdu_data,
                'english_version': english_data
            })
        
        return {
            'email': email,
            'total_records': len(histories),
            'histories': histories
        }
        
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve medical histories: {str(e)}"
        )


app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/index")
async def serve_index():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
