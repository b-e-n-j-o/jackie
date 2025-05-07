from fastapi import FastAPI, Form, Request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
import redis
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import json
from supabase import create_client
import requests
import logging
import sys
import time
import threading

from chat_logic import process_message

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configuration Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")

# Configuration Redis
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_TTL = int(os.getenv("REDIS_TTL", 20))  # 5 minutes par défaut

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL_DEV")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_DEV")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuration Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")  # Format: whatsapp:+1234567890

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
    raise

def send_whatsapp_message(to_number: str, message: str) -> dict:
    """Envoie un message WhatsApp via Twilio"""
    try:
        # Formatage du numéro pour Twilio
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
        
        url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json'
        data = {
            'To': to_number,
            'From': TWILIO_PHONE_NUMBER,
            'Body': message
        }
        
        response = requests.post(
            url,
            data=data,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        
        if response.status_code == 201:
            logger.info(f"Message WhatsApp envoyé avec succès à {to_number}")
            return {
                "status": "success",
                "message_sid": response.json().get("sid")
            }
        else:
            logger.error(f"Erreur lors de l'envoi du message WhatsApp: {response.text}")
            return {
                "status": "error",
                "error": response.text
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message WhatsApp: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def get_active_supabase_session(phone_number: str):
    """Récupère la session active Supabase pour ce numéro, ou None."""
    try:
        response = supabase.table("sessions").select("*") \
            .eq("phone_number", phone_number) \
            .eq("status", "active") \
            .order("start_time", desc=True) \
            .limit(1) \
            .execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la session active Supabase: {str(e)}")
        return None

def create_supabase_session(session: dict, user_id: str):
    """Crée une nouvelle session dans Supabase avec status 'active'."""
    import time
    session_id = f"{user_id}_{int(time.time())}"
    try:
        session_data = {
            "id": session_id,
            "user_id": user_id,
            "phone_number": session["phone_number"],
            "start_time": session["created_at"],
            "last_activity": session["last_activity"],
            "messages": json.dumps(session["messages"]),
            "status": "active"
        }
        result = supabase.table("sessions").insert(session_data).execute()
        logger.info(f"Session créée dans Supabase pour {session['phone_number']}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la création de la session dans Supabase: {str(e)}")
        return None

def close_supabase_session(phone_number: str, session_data: dict):
    """Met à jour la session active Supabase en la passant à 'closed' et en mettant à jour les messages et end_time."""
    try:
        active_session = get_active_supabase_session(phone_number)
        if not active_session:
            logger.warning(f"Aucune session active à clôturer pour {phone_number}")
            return None
            
        # Import chat_logic pour fermer la session correctement
        from chat_logic import close_session
        
        # Si on a session_id, appeler close_session de chat_logic
        session_id = session_data.get("session_id")
        if session_id:
            try:
                close_session(session_id)
                logger.info(f"Session {session_id} fermée via chat_logic")
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture via chat_logic: {str(e)}")
        
        # Sinon, on fait un update manuel
        updates = {
            "end_time": datetime.now().isoformat(),
            "last_activity": session_data["last_activity"],
            "messages": json.dumps(session_data["messages"]),
            "status": "closed"
        }
        result = supabase.table("sessions").update(updates).eq("id", active_session["id"]).execute()
        logger.info(f"Session clôturée dans Supabase pour {phone_number}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la clôture de la session dans Supabase: {str(e)}")
        return None

def get_or_create_session(phone_number: str) -> dict:
    """Récupère une session existante ou en crée une nouvelle."""
    session_key = f"session:{phone_number}"
    logger.info(f"Tentative de récupération/création de session pour {phone_number}")
    try:
        if redis_client.exists(session_key):
            session_data = json.loads(redis_client.get(session_key))
            redis_client.expire(session_key, REDIS_TTL)
            logger.info(f"Session existante récupérée pour {phone_number}")
            return session_data
            
        # Nouvelle session
        from chat_logic import get_user_id
        user_id = get_user_id(phone_number)
        if not user_id:
            logger.error(f"Utilisateur non trouvé pour le numéro {phone_number}")
            # Renvoyer une session minimale
            return {
                "phone_number": phone_number,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            
        # Vérifier s'il y a déjà une session active dans Supabase
        active_session = get_active_supabase_session(phone_number)
        if active_session:
            # Récupérer la session existante de Supabase
            logger.info(f"Session active trouvée dans Supabase pour {phone_number}")
            session_data = {
                "phone_number": phone_number,
                "user_id": user_id,
                "session_id": active_session["id"],
                "messages": json.loads(active_session["messages"]) if active_session.get("messages") else [],
                "created_at": active_session["start_time"],
                "last_activity": active_session["last_activity"]
            }
        else:
            # Créer une nouvelle session
            session_id = f"{user_id}_{int(time.time())}"
            session_data = {
                "phone_number": phone_number,
                "user_id": user_id,
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            
            # Créer la session dans Supabase
            try:
                session_info = {
                    "id": session_id,
                    "user_id": user_id,
                    "phone_number": phone_number,
                    "start_time": session_data["created_at"],
                    "last_activity": session_data["last_activity"],
                    "messages": json.dumps([]),
                    "status": "active"
                }
                supabase.table("sessions").insert(session_info).execute()
                logger.info(f"Nouvelle session créée dans Supabase: {session_id}")
            except Exception as e:
                logger.error(f"Erreur lors de la création de la session dans Supabase: {str(e)}")
                
        # Stocker en Redis
        redis_client.setex(session_key, REDIS_TTL, json.dumps(session_data))
        logger.info(f"Session créée et stockée dans Redis pour {phone_number}")
        return session_data
        
    except Exception as e:
        logger.error(f"Erreur lors de la gestion de la session pour {phone_number}: {str(e)}")
        # Renvoyer une session minimale en cas d'erreur
        return {
            "phone_number": phone_number,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    logger.info("Début du traitement du webhook")
    form = await request.form()
    logger.info(f"Form data reçu: {form}")
    user_number = form["From"].split(":")[-1]
    user_message = form["Body"]
    
    logger.info(f"Nouveau message reçu de {user_number}: {user_message}")
    
    try:
        # Récupérer ou créer la session
        logger.info(f"Tentative de récupération/création de session pour {user_number}")
        session_data = get_or_create_session(user_number)
        logger.info(f"Session récupérée/créée: {session_data.get('session_id', 'No ID')}")
        
        # Traiter le message avec chat_logic
        process_result = await process_message(session_data, user_message, user_number)
        
        if not process_result["success"]:
            logger.error(f"Erreur de traitement: {process_result.get('response', 'Unknown error')}")
            reply = "Désolé, une erreur s'est produite lors du traitement de votre message."
        else:
            reply = process_result["response"]
            session_data = process_result["session_data"]
        
        # Mettre à jour la session dans Redis
        session_key = f"session:{user_number}"
        session_data["last_activity"] = datetime.now().isoformat()
        redis_client.setex(session_key, REDIS_TTL, json.dumps(session_data))
        logger.info(f"Session mise à jour dans Redis pour {user_number}")
        
        # Mettre à jour la session active dans Supabase
        if session_data.get("session_id"):
            try:
                supabase.table("sessions").update({
                    "last_activity": session_data["last_activity"],
                    "messages": json.dumps(session_data["messages"])
                }).eq("id", session_data["session_id"]).execute()
                logger.info(f"Session mise à jour dans Supabase: {session_data['session_id']}")
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour Supabase: {str(e)}")
        
        # Vérifier si un appel a été initié
        if process_result.get("call_initiated", False):
            logger.info(f"Appel initié pour {user_number}, pas besoin d'envoyer un message WhatsApp")
            # Ne pas envoyer de message WhatsApp car c'est géré dans chat_logic
        else:
            # Envoyer la réponse via Twilio si pas déjà fait
            send_result = send_whatsapp_message(user_number, reply)
            logger.info(f"Résultat de l'envoi du message via Twilio: {send_result}")
        
        # Réponse pour Twilio
        twilio_response = MessagingResponse()
        twilio_response.message(reply)
        return str(twilio_response)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook pour {user_number}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Réponse d'erreur générique
        twilio_response = MessagingResponse()
        twilio_response.message("Désolé, une erreur s'est produite lors du traitement de votre message.")
        return str(twilio_response)

@app.get("/monitor/sessions")
def monitor_sessions():
    """Endpoint pour surveiller les sessions actives"""
    try:
        active_sessions = []
        for key in redis_client.keys("session:*"):
            session_data = json.loads(redis_client.get(key))
            active_sessions.append({
                "phone_number": session_data["phone_number"],
                "message_count": len(session_data.get("messages", [])),
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"],
                "session_id": session_data.get("session_id", "unknown"),
                "ttl": redis_client.ttl(key)
            })
        
        logger.info(f"Récupération de {len(active_sessions)} sessions actives")
        return {"active_sessions": active_sessions}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {str(e)}")
        return {"error": str(e)}

@app.get("/monitor/vapi")
def monitor_vapi_calls():
    """Endpoint pour surveiller les appels VAPI récents"""
    try:
        from chat_logic import last_vapi_call_info
        return {"last_vapi_call": last_vapi_call_info}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données VAPI: {str(e)}")
        return {"error": str(e)}

def close_expired_sessions():
    """Ferme toutes les sessions inactives dans Redis et Supabase"""
    try:
        logger.info("Vérification des sessions expirées")
        logger.info(f"Heure serveur locale: {datetime.now().isoformat()}")
        now = datetime.now()
        count = 0
        
        # Fermer les sessions Redis expirées
        for key in redis_client.keys("session:*"):
            try:
                session_data = json.loads(redis_client.get(key))
                phone_number = session_data.get("phone_number")
                last_activity_str = session_data.get("last_activity")
                session_id = session_data.get("session_id")
                
                if not phone_number or not last_activity_str or not session_id:
                    continue
                    
                # Standardiser le format de date - SANS timezone
                try:
                    # Retirer toute information de timezone pour correspondre à l'heure locale
                    if 'Z' in last_activity_str:
                        last_activity_str = last_activity_str.replace('Z', '')
                    if '+' in last_activity_str:
                        last_activity_str = last_activity_str.split('+')[0]
                    
                    last_activity = datetime.fromisoformat(last_activity_str)
                    
                    # Calculer la différence brute, sans tenir compte des fuseaux horaires
                    time_diff = (now - last_activity).total_seconds()
                    
                    logger.info(f"Session {phone_number} - Dernière activité: {last_activity.isoformat()}")
                    logger.info(f"Différence de temps: {time_diff}s (Limite: {REDIS_TTL}s)")
                    
                    if time_diff > REDIS_TTL:
                        logger.info(f"Fermeture de la session {phone_number} après {time_diff} secondes d'inactivité")
                        
                        # Utiliser la fonction close_session de chat_logic
                        from chat_logic import close_session
                        close_session(session_id)
                        
                        # Supprimer de Redis
                        redis_client.delete(key)
                        count += 1
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la date: {last_activity_str}")
                    logger.error(str(e))
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la session {key}: {str(e)}")
                
        # Fermer les sessions Supabase actives mais inactives
        try:
            response = supabase.table("sessions").select("id, phone_number, last_activity").eq("status", "active").execute()
            logger.info(f"Sessions actives trouvées dans Supabase: {len(response.data)}")
            
            for session in response.data:
                session_id = session.get("id")
                phone_number = session.get("phone_number")
                last_activity_str = session.get("last_activity")
                
                if not last_activity_str:
                    continue
                
                try:
                    # Retirer toute information de timezone pour correspondre à l'heure locale
                    if 'Z' in last_activity_str:
                        last_activity_str = last_activity_str.replace('Z', '')
                    if '+' in last_activity_str:
                        last_activity_str = last_activity_str.split('+')[0]
                    
                    last_activity = datetime.fromisoformat(last_activity_str)
                    
                    # Calculer la différence brute
                    time_diff = (now - last_activity).total_seconds()
                    
                    logger.info(f"Session Supabase {session_id} ({phone_number}) - Dernière activité: {last_activity.isoformat()}")
                    logger.info(f"Différence de temps: {time_diff}s (Limite: {REDIS_TTL}s)")
                    
                    if time_diff > REDIS_TTL:
                        logger.info(f"Fermeture de la session Supabase {session_id} après {time_diff} secondes")
                        
                        # Utiliser la fonction close_session de chat_logic si possible
                        try:
                            from chat_logic import close_session
                            close_session(session_id)
                        except Exception as e:
                            logger.error(f"Erreur lors de l'appel à close_session de chat_logic: {str(e)}")
                            # Fallback: mise à jour directe si close_session échoue
                            supabase.table("sessions").update({
                                "end_time": now.isoformat(),
                                "status": "closed"
                            }).eq("id", session_id).execute()
                        
                        count += 1
                except Exception as e:
                    logger.error(f"Erreur lors de la fermeture de la session Supabase {session_id}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des sessions Supabase: {str(e)}")
            
        logger.info(f"{count} sessions fermées automatiquement")
        return count
    except Exception as e:
        logger.error(f"Erreur lors de la clôture automatique des sessions: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

def background_session_cleaner():
    """Fonction exécutée en arrière-plan pour nettoyer les sessions expirées"""
    while True:
        try:
            close_expired_sessions()
        except Exception as e:
            logger.error(f"Erreur dans le background cleaner: {str(e)}")
        time.sleep(60)  # Vérifier toutes les minutes

@app.on_event("startup")
def start_background_cleaner():
    """Démarre le processus de nettoyage des sessions en arrière-plan"""
    threading.Thread(target=background_session_cleaner, daemon=True).start()
    logger.info("Processus de nettoyage des sessions démarré")