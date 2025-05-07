from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse
import os
import logging
import json
import requests
from datetime import datetime, timezone
import time
import threading
from dotenv import load_dotenv
import redis

# Import des fonctions du module chat_logic
from app.chat import (
    process_message, get_user_id, close_session, 
    send_whatsapp_message, last_vapi_call_info
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configuration Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_TTL = int(os.getenv("REDIS_TTL", 1800))  # 30 minutes par défaut

# Connexion Redis
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ssl=True,
        decode_responses=True
    )
    # Test de connexion
    redis_client.ping()
    logger.info(f"Connexion Redis établie avec succès sur {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.error(f"Erreur de connexion Redis: {str(e)}")
    redis_client = None
    logger.warning("Continuing without Redis - using in-memory session handling")

def get_or_create_session(phone_number: str) -> dict:
    """Récupère une session existante ou en crée une nouvelle."""
    session_key = f"session:{phone_number}"
    logger.info(f"Tentative de récupération/création de session pour {phone_number}")
    try:
        if redis_client and redis_client.exists(session_key):
            session_data = json.loads(redis_client.get(session_key))
            redis_client.expire(session_key, REDIS_TTL)
            logger.info(f"Session existante récupérée pour {phone_number}")
            return session_data
            
        # Nouvelle session
        user_id = get_user_id(phone_number)
        if not user_id:
            logger.error(f"Utilisateur non trouvé pour le numéro {phone_number}")
            # Renvoyer une session minimale
            return {
                "phone_number": phone_number,
                "messages": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat()
            }
            
        # Créer une nouvelle session
        session_id = f"{user_id}_{int(time.time())}"
        session_data = {
            "phone_number": phone_number,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        # Stocker en Redis
        if redis_client:
            redis_client.setex(session_key, REDIS_TTL, json.dumps(session_data))
            logger.info(f"Session créée et stockée dans Redis pour {phone_number}")
        
        return session_data
        
    except Exception as e:
        logger.error(f"Erreur lors de la gestion de la session pour {phone_number}: {str(e)}")
        # Renvoyer une session minimale en cas d'erreur
        return {
            "phone_number": phone_number,
            "messages": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    logger.info("Début du traitement du webhook")
    form = await request.form()
    logger.info(f"Form data reçu: {dict(form)}")
    
    try:
        # Extraction des données principales
        user_number = form.get("From", "").split(":")[-1] if "From" in form else ""
        user_message = form.get("Body", "")
        message_sid = form.get("SmsMessageSid", "")
        
        if not user_number or not user_message:
            logger.error("Données manquantes dans la requête")
            return PlainTextResponse(content="", status_code=200)
        
        logger.info(f"Nouveau message reçu de {user_number}: {user_message[:50]}...")
        
        # Récupérer ou créer la session
        logger.info(f"Tentative de récupération/création de session pour {user_number}")
        session_data = get_or_create_session(user_number)
        logger.info(f"Session récupérée/créée: {session_data.get('session_id', 'No ID')}")
        
        # Traiter le message avec chat_logic
        process_result = await process_message(
            session_data=session_data, 
            message=user_message, 
            phone_number=user_number,
            redis_client=redis_client
        )
        
        if not process_result["success"]:
            logger.error(f"Erreur de traitement: {process_result.get('response', 'Unknown error')}")
            return PlainTextResponse(content="", status_code=200)
            
        response_text = process_result["response"]
        session_data = process_result["session_data"]
        
        # Mettre à jour la session dans Redis
        if redis_client:
            session_key = f"session:{user_number}"
            session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
            redis_client.setex(session_key, REDIS_TTL, json.dumps(session_data))
            logger.info(f"Session mise à jour dans Redis pour {user_number}")
        
        # Vérifier si un appel a été initié
        if process_result.get("call_initiated", False):
            logger.info(f"Appel initié pour {user_number}, envoi du message de confirmation")
            # Envoyer explicitement le message WhatsApp de confirmation
            send_result = send_whatsapp_message(user_number, process_result["response"])
            logger.info(f"Résultat de l'envoi du message de confirmation: {send_result}")
        else:
            # Envoyer le message uniquement si un appel n'a pas été initié
            send_result = send_whatsapp_message(user_number, response_text)
            logger.info(f"Envoi du message WhatsApp: {send_result}")
        
        # Réponse vide pour Twilio pour éviter un doublon
        return PlainTextResponse(content="", status_code=200)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Réponse d'erreur générique
        return PlainTextResponse(content="", status_code=200)

@app.get("/monitor/sessions")
def monitor_sessions():
    """Endpoint pour surveiller les sessions actives"""
    try:
        if not redis_client:
            return {"error": "Redis not connected", "active_sessions": []}
            
        active_sessions = []
        for key in redis_client.keys("session:*"):
            session_data = json.loads(redis_client.get(key))
            active_sessions.append({
                "phone_number": session_data.get("phone_number", "Unknown"),
                "message_count": len(session_data.get("messages", [])),
                "created_at": session_data.get("created_at", "Unknown"),
                "last_activity": session_data.get("last_activity", "Unknown"),
                "session_id": session_data.get("session_id", "unknown"),
                "ttl": redis_client.ttl(key)
            })
        
        logger.info(f"Récupération de {len(active_sessions)} sessions actives")
        return {"active_sessions": active_sessions}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {str(e)}")
        return {"error": str(e), "active_sessions": []}

@app.get("/monitor/vapi")
def monitor_vapi_calls():
    """Endpoint pour surveiller les appels VAPI récents"""
    try:
        return {"last_vapi_call": last_vapi_call_info}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données VAPI: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Endpoint pour vérifier que l'application est en ligne"""
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status
    }

def close_expired_sessions():
    """Ferme toutes les sessions inactives dans Redis et Supabase"""
    try:
        logger.info("Vérification des sessions expirées")
        if not redis_client:
            logger.info("Redis non connecté, impossible de vérifier les sessions expirées")
            return 0
            
        count = 0
        for key in redis_client.keys("session:*"):
            try:
                session_data = json.loads(redis_client.get(key))
                session_id = session_data.get("session_id")
                last_activity_str = session_data.get("last_activity")
                
                if not session_id or not last_activity_str:
                    continue
                    
                # Calcul du temps d'inactivité
                try:
                    if 'Z' in last_activity_str:
                        last_activity_str = last_activity_str.replace('Z', '+00:00')
                    
                    last_activity = datetime.fromisoformat(last_activity_str)
                    current_time = datetime.now(timezone.utc)
                    
                    # S'assurer que last_activity a un fuseau horaire
                    if last_activity.tzinfo is None:
                        last_activity = last_activity.replace(tzinfo=timezone.utc)
                    
                    time_diff = (current_time - last_activity).total_seconds()
                    
                    if time_diff > REDIS_TTL:
                        logger.info(f"Fermeture de la session {session_id} après {time_diff} secondes d'inactivité")
                        close_session(session_id, redis_client)
                        count += 1
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la date: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la session {key}: {str(e)}")
                
        logger.info(f"{count} sessions fermées automatiquement")
        return count
    except Exception as e:
        logger.error(f"Erreur lors de la clôture automatique des sessions: {str(e)}")
        return 0

def background_session_cleaner():
    """Fonction exécutée en arrière-plan pour nettoyer les sessions expirées"""
    while True:
        try:
            close_expired_sessions()
        except Exception as e:
            logger.error(f"Erreur dans le background cleaner: {str(e)}")
        time.sleep(300)  # Vérifier toutes les 5 minutes

@app.on_event("startup")
def start_background_cleaner():
    """Démarre le processus de nettoyage des sessions en arrière-plan"""
    threading.Thread(target=background_session_cleaner, daemon=True).start()
    logger.info("Processus de nettoyage des sessions démarré")