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
REDIS_TTL = int(os.getenv("REDIS_TTL", 20))  # 20 secondes par défaut

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

# Initialisation du LLM
llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    model=AZURE_OPENAI_MODEL,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-07-01-preview"
)

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
            print(f"Message WhatsApp envoyé avec succès à {to_number}")
            return {
                "status": "success",
                "message_sid": response.json().get("sid")
            }
        else:
            print(f"Erreur lors de l'envoi du message WhatsApp: {response.text}")
            return {
                "status": "error",
                "error": response.text
            }
            
    except Exception as e:
        print(f"Erreur lors de l'envoi du message WhatsApp: {str(e)}")
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

def close_supabase_session(phone_number: str, session: dict):
    """Met à jour la session active Supabase en la passant à 'closed' et en mettant à jour les messages et end_time."""
    try:
        active_session = get_active_supabase_session(phone_number)
        if not active_session:
            logger.warning(f"Aucune session active à clôturer pour {phone_number}")
            return None
        updates = {
            "end_time": datetime.now().isoformat(),
            "last_activity": session["last_activity"],
            "messages": json.dumps(session["messages"]),
            "status": "closed"
        }
        result = supabase.table("sessions").update(updates).eq("id", active_session["id"]).execute()
        logger.info(f"Session clôturée dans Supabase pour {phone_number}")
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la clôture de la session dans Supabase: {str(e)}")
        return None

def get_or_create_session(phone_number: str) -> dict:
    session_key = f"session:{phone_number}"
    logger.info(f"Tentative de récupération/création de session pour {phone_number}")
    try:
        if redis_client.exists(session_key):
            session_data = json.loads(redis_client.get(session_key))
            redis_client.expire(session_key, REDIS_TTL)
            logger.info(f"Session existante récupérée pour {phone_number}")
            return session_data
        # Nouvelle session : vérifier s'il y a déjà une session active dans Supabase
        if not get_active_supabase_session(phone_number):
            user_id = get_user_id_by_phone(phone_number)
            if not user_id:
                # Gérer le cas où l'utilisateur n'existe pas
                return None
            new_session = {
                "phone_number": phone_number,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            redis_client.setex(session_key, REDIS_TTL, json.dumps(new_session))
            logger.info(f"Nouvelle session créée pour {phone_number}")
            create_supabase_session(new_session, user_id)
            return new_session
        else:
            # Il y a déjà une session active Supabase, mais pas en cache Redis (cas rare)
            # On peut la recharger ou en créer une nouvelle selon la logique métier
            # Ici, on recharge la session Supabase en mémoire Redis
            active = get_active_supabase_session(phone_number)
            session = {
                "phone_number": phone_number,
                "messages": json.loads(active["messages"]),
                "created_at": active["start_time"],
                "last_activity": active["last_activity"]
            }
            redis_client.setex(session_key, REDIS_TTL, json.dumps(session))
            return session
    except Exception as e:
        logger.error(f"Erreur lors de la gestion de la session pour {phone_number}: {str(e)}")
        raise

def save_session_to_supabase(session: dict):
    """Sauvegarde une session dans Supabase"""
    try:
        session_data = {
            "phone_number": session["phone_number"],
            "messages": json.dumps(session["messages"]),
            "created_at": session["created_at"],
            "ended_at": datetime.now().isoformat(),
            "duration": (datetime.fromisoformat(session["last_activity"]) - 
                       datetime.fromisoformat(session["created_at"])).total_seconds()
        }
        
        result = supabase.table("sessions").insert(session_data).execute()
        logger.info(f"Session sauvegardée dans Supabase pour {session['phone_number']}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de la session dans Supabase: {str(e)}")
        raise

def get_user_id_by_phone(phone_number: str) -> str:
    """Récupère le user_id à partir du numéro de téléphone."""
    try:
        response = supabase.table("users").select("id").eq("phone_number", phone_number).limit(1).execute()
        if response.data:
            return response.data[0]["id"]
        else:
            logger.error(f"Aucun user_id trouvé pour le numéro {phone_number}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du user_id: {str(e)}")
        return None

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    user_number = form["From"].split(":")[-1]
    user_message = form["Body"]
    
    logger.info(f"Nouveau message reçu de {user_number}: {user_message}")
    
    try:
        # Récupérer ou créer la session
        session = get_or_create_session(user_number)
        
        # Ajouter le message utilisateur à la session
        session["messages"].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Générer la réponse avec Azure OpenAI
        logger.info(f"Génération de réponse avec Azure OpenAI pour {user_number}")
        response = llm([HumanMessage(content=user_message)])
        reply = response.content
        logger.info(f"Réponse générée pour {user_number}: {reply[:100]}...")
        
        # Ajouter la réponse à la session
        session["messages"].append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mettre à jour la session dans Redis
        session_key = f"session:{user_number}"
        session["last_activity"] = datetime.now(timezone.utc).isoformat()
        redis_client.setex(session_key, REDIS_TTL, json.dumps(session))
        logger.info(f"Session mise à jour dans Redis pour {user_number}")
        
        # Mettre à jour la session active dans Supabase
        active_session = get_active_supabase_session(user_number)
        if active_session:
            supabase.table("sessions").update({
                "last_activity": session["last_activity"],
                "messages": json.dumps(session["messages"])
            }).eq("id", active_session["id"]).execute()
        
        # Si la session a expiré (clé absente), on clôture la session Supabase
        if not redis_client.exists(session_key):
            logger.info(f"Session expirée pour {user_number}, clôture dans Supabase")
            close_supabase_session(user_number, session)
        
        # Envoyer la réponse via Twilio
        send_result = send_whatsapp_message(user_number, reply)
        logger.info(f"Résultat de l'envoi du message via Twilio: {send_result['status']}")
        
        # Ajouter le statut d'envoi à la session
        session["messages"][-1]["send_status"] = send_result["status"]
        if send_result["status"] == "success":
            session["messages"][-1]["message_sid"] = send_result["message_sid"]
        
        # Mettre à jour la session avec le statut d'envoi
        redis_client.setex(session_key, REDIS_TTL, json.dumps(session))
        
        # Retourner la réponse Twilio
        twilio_response = MessagingResponse()
        twilio_response.message(reply)
        return str(twilio_response)
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook pour {user_number}: {str(e)}")
        raise

@app.get("/monitor/sessions")
def monitor_sessions():
    """Endpoint pour surveiller les sessions actives"""
    try:
        active_sessions = []
        for key in redis_client.keys("session:*"):
            session_data = json.loads(redis_client.get(key))
            active_sessions.append({
                "phone_number": session_data["phone_number"],
                "message_count": len(session_data["messages"]),
                "created_at": session_data["created_at"],
                "last_activity": session_data["last_activity"]
            })
        
        logger.info(f"Récupération de {len(active_sessions)} sessions actives")
        return {"active_sessions": active_sessions}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sessions: {str(e)}")
        raise

def background_session_cleaner():
    while True:
        try:
            close_all_expired_sessions()
        except Exception as e:
            logger.error(f"Erreur dans le background cleaner: {str(e)}")
        time.sleep(20)  # ou la fréquence souhaitée

@app.on_event("startup")
def start_background_cleaner():
    threading.Thread(target=background_session_cleaner, daemon=True).start()

def close_all_expired_sessions():
    """Ferme toutes les sessions actives dont la dernière activité est supérieure au TTL."""
    try:
        now = datetime.now(timezone.utc)  # <-- UTC et aware
        ttl_seconds = REDIS_TTL
        # Récupérer toutes les sessions actives
        response = supabase.table("sessions").select("id, last_activity").eq("status", "active").execute()
        closed_count = 0
        for session in response.data:
            last_activity = session.get("last_activity")
            if last_activity:
                try:
                    last_dt = datetime.fromisoformat(last_activity)
                    if last_dt.tzinfo is None:
                        # Si la date est naive, on la force en UTC
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if (now - last_dt).total_seconds() > ttl_seconds:
                    # Clôturer la session
                    updates = {
                        "end_time": now.isoformat(),
                        "status": "closed"
                    }
                    supabase.table("sessions").update(updates).eq("id", session["id"]).execute()
                    closed_count += 1
        logger.info(f"{closed_count} sessions clôturées automatiquement.")
        return closed_count
    except Exception as e:
        logger.error(f"Erreur lors de la clôture automatique des sessions: {str(e)}")
        return 0
