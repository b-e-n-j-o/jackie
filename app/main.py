from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from app.chat_logic import handle_user_message, check_inactive_sessions, chat_history_store, is_session_inactive, last_activity_times
import os
import threading
import time
from typing import Dict
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage

# Variable globale pour stocker la dernière requête Twilio
LAST_TWILIO_REQUEST = None

app = FastAPI(
    title="Jackie API",
    description="API de chat pour Jackie, le connecteur social IA",
    version="1.0.0"
)

# Rate limiting
RATE_LIMIT: Dict[str, list] = {}
MAX_REQUESTS = 100  # Nombre maximum de requêtes par numéro
TIME_WINDOW = 3600  # Période en secondes (1 heure)

def check_rate_limit(phone_number: str):
    now = datetime.now()
    if phone_number not in RATE_LIMIT:
        RATE_LIMIT[phone_number] = []
    
    # Nettoyer les anciennes requêtes
    RATE_LIMIT[phone_number] = [
        timestamp for timestamp in RATE_LIMIT[phone_number]
        if timestamp > now - timedelta(seconds=TIME_WINDOW)
    ]
    
    if len(RATE_LIMIT[phone_number]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes. Veuillez réessayer plus tard."
        )
    
    RATE_LIMIT[phone_number].append(now)

@app.post("/webhook/twilio")
async def twilio_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
):
    try:
        # Récupérer TOUTES les données brutes
        form_data = await request.form()
        headers = dict(request.headers)
        
        # Stocker ces données pour diagnostic
        global LAST_TWILIO_REQUEST
        LAST_TWILIO_REQUEST = {
            "timestamp": datetime.now().isoformat(),
            "headers": headers,
            "form_data": {k: v for k, v in form_data.items()},
            "raw_from": form_data.get("From", "Non trouvé"),
            "raw_body": form_data.get("Body", "Non trouvé")
        }
        
        # Nettoyage du numéro de téléphone
        phone_number = From.replace('whatsapp:', '')
        
        # Vérification du rate limit
        check_rate_limit(phone_number)
        
        # Validation de la longueur du message
        if not Body or len(Body) > 2000:
            return JSONResponse(
                status_code=400,
                content={"error": "Le message doit faire entre 1 et 2000 caractères"}
            )
        
        # Protection contre les caractères dangereux
        if any(char in Body for char in ['<', '>', ';', '--']):
            return JSONResponse(
                status_code=400,
                content={"error": "Message contient des caractères non autorisés"}
            )
        
        # Traitement du message via la logique existante
        response = await handle_user_message(phone_number, Body)
        
        return {"message": response}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du traitement du message: {str(e)}"}
        )

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        
        # Validation manuelle des champs
        phone_number = data.get("phone_number")
        message = data.get("message")
        
        # Validation du numéro de téléphone
        if not phone_number or not isinstance(phone_number, str) or not phone_number.startswith('+'):
            raise HTTPException(
                status_code=400,
                detail="Format de numéro de téléphone invalide"
            )
        
        # Validation du message
        if not message or not isinstance(message, str) or len(message) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Le message doit faire entre 1 et 2000 caractères"
            )
        
        # Vérification du rate limit
        check_rate_limit(phone_number)
        
        # Protection contre les caractères dangereux
        if any(char in message for char in ['<', '>', ';', '--']):
            raise HTTPException(
                status_code=400,
                detail="Message contient des caractères non autorisés"
            )
        
        response = await handle_user_message(phone_number, message)
        return {"response": response}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement du message: {str(e)}"
        )

def background_session_checker():
    while True:
        time.sleep(10)
        try:
            check_inactive_sessions()
        except Exception as e:
            print(f"Erreur dans le background checker: {e}")

@app.on_event("startup")
def start_background_tasks():
    thread = threading.Thread(target=background_session_checker, daemon=True)
    thread.start()

@app.get("/monitor/active-sessions")
def monitor_active_sessions():
    # Liste des sessions actives
    active_sessions = []
    for session_id, history in chat_history_store.items():
        if not is_session_inactive(session_id):
            active_sessions.append({
                "session_id": session_id,
                "phone_number": history.phone_number,
                "messages_count": len(history.session_messages),
                "messages": [
                    {
                        "type": "user" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content
                    }
                    for msg in history.session_messages
                ],
                "last_activity": last_activity_times[session_id].isoformat() if session_id in last_activity_times else None
            })
    
    return {
        "count": len(active_sessions),
        "sessions": active_sessions
    } 

@app.get("/monitor/incoming-messages")
def monitor_incoming_messages():
    # Liste des requêtes récentes
    recent_requests = []
    
    for phone, timestamps in RATE_LIMIT.items():
        if timestamps:
            recent_requests.append({
                "phone_number": phone,
                "timestamp": timestamps[-1].isoformat() if timestamps else None,
                "count_last_hour": len(timestamps)
            })
    
    return {
        "count": len(recent_requests),
        "requests": recent_requests
    }

@app.get("/monitor/twilio-debug")
def monitor_twilio_debug():
    return {
        "last_request": LAST_TWILIO_REQUEST if LAST_TWILIO_REQUEST is not None else "Aucune requête reçue"
    }