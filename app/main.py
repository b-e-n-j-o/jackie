from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from app.chat_logic import handle_user_message, check_inactive_sessions
import os
import threading
import time
from typing import Dict
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from supabase import create_client, Client
from collections import OrderedDict
import traceback

# Configuration Supabase
supabase: Client = create_client(
   os.getenv("SUPABASE_URL_DEV", ""),
   os.getenv("SUPABASE_KEY_DEV", "")
)

SESSION_TIMEOUT = 3600  # 1 heure

# Stockage des sessions en mémoire avec LRU Cache
class SessionCache:
   def __init__(self, max_size=1000, ttl=300):  # 5 minutes TTL
       self.cache = OrderedDict()
       self.max_size = max_size
       self.ttl = ttl
       self.lock = threading.Lock()
   
   def get(self, key: str) -> dict:
       with self.lock:
           if key in self.cache:
               session = self.cache[key]
               if (datetime.now() - session["last_activity"]).total_seconds() < self.ttl:
                   # Mettre à jour la position dans le cache (LRU)
                   self.cache.move_to_end(key)
                   return session
               else:
                   # Session expirée, la sauvegarder dans Supabase
                   self._save_to_supabase(key, session)
                   del self.cache[key]
           return None
   
   def set(self, key: str, value: dict):
       with self.lock:
           if len(self.cache) >= self.max_size:
               # Supprimer la session la plus ancienne
               oldest_key, oldest_session = self.cache.popitem(last=False)
               self._save_to_supabase(oldest_key, oldest_session)
           
           value["last_activity"] = datetime.now()
           self.cache[key] = value
           self.cache.move_to_end(key)
   
   def _save_to_supabase(self, phone_number: str, session: dict):
       try:
           supabase.table("chat_sessions").insert({
               "phone_number": phone_number,
               "messages": session["messages"],
               "created_at": session["created_at"],
               "ended_at": datetime.now().isoformat()
           }).execute()
       except Exception as e:
           print(f"Erreur lors de la sauvegarde dans Supabase: {e}")
   
   def cleanup(self):
       """Nettoie les sessions expirées"""
       with self.lock:
           now = datetime.now()
           expired_keys = [
               key for key, session in self.cache.items()
               if (now - session["last_activity"]).total_seconds() >= self.ttl
           ]
           for key in expired_keys:
               self._save_to_supabase(key, self.cache[key])
               del self.cache[key]

# Initialisation du cache de sessions
session_cache = SessionCache()

def get_or_create_session(phone_number: str) -> dict:
   """Récupère ou crée une session pour un numéro de téléphone"""
   session = session_cache.get(phone_number)
   if session:
       return session
   
   # Créer une nouvelle session
   new_session = {
       "messages": [],
       "created_at": datetime.now().isoformat(),
       "last_activity": datetime.now()
   }
   session_cache.set(phone_number, new_session)
   return new_session

def background_cleanup():
   """Nettoyage périodique des sessions expirées"""
   while True:
       time.sleep(60)  # Vérifier toutes les minutes
       session_cache.cleanup()

# Démarrer le nettoyage en arrière-plan
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

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
           "form_data": {k: str(v) for k, v in form_data.items()},
           "raw_from": form_data.get("From", "Non trouvé"),
           "raw_body": form_data.get("Body", "Non trouvé")
       }
       
       # Nettoyage du numéro de téléphone
       phone_number = From.replace('whatsapp:', '').strip()
       if not phone_number.startswith('+'):
           phone_number = '+' + phone_number
           
       print(f"Numéro nettoyé: {phone_number}")
       
       # Récupération ou création de la session
       session = get_or_create_session(phone_number)
       
       # Ajout du message utilisateur
       session["messages"].append({
           "role": "user",
           "content": Body,
           "timestamp": datetime.now().isoformat()
       })
       
       # Mise à jour de la session
       session_cache.set(phone_number, session)
       
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
       
       # Ajout de la réponse à la session
       session["messages"].append({
           "role": "assistant",
           "content": response,
           "timestamp": datetime.now().isoformat()
       })
       
       # Mise à jour finale de la session
       session_cache.set(phone_number, session)
       
       return {"message": response}
       
   except Exception as e:
       traceback.print_exc()
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
   """Endpoint pour voir les sessions actives"""
   active_sessions = []
   for phone_number, session in session_cache.cache.items():
       active_sessions.append({
           "phone_number": phone_number,
           "message_count": len(session["messages"]),
           "last_activity": session["last_activity"].isoformat(),
           "created_at": session["created_at"],
           "ttl_remaining": session_cache.ttl - (datetime.now() - session["last_activity"]).total_seconds()
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