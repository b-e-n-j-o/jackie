import os
import json
import logging
import time
import uuid
import random
import requests
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dotenv import load_dotenv

# Import OpenAI directement
from openai import AzureOpenAI

# Import core LangChain components
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import depuis notre app
from supabase import create_client, Client

from pydantic.v1 import BaseModel, Field


load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration Azure OpenAI
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Configuration Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Configuration VAPI
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID")
VAPI_API_URL = os.getenv("VAPI_API_URL")
VAPI_API_KEY = os.getenv("VAPI_API_KEY")

# Configuration des sessions
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "30"))  # 30 secondes par défaut
last_activity_times: Dict[str, datetime] = {}
chat_history_store: Dict[str, BaseChatMessageHistory] = {}

# Stockage des informations du dernier appel VAPI pour monitoring
last_vapi_call_info = {
    "timestamp": None,
    "phone_number": None,
    "name": None,
    "bio_preview": None,
    "bio_length": 0,
    "status_code": None,
    "response": None
}

# Initialisation du client OpenAI
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL_DEV")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_DEV")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Variables d'environnement SUPABASE_URL et SUPABASE_KEY requises")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SupabaseMessageHistory(BaseChatMessageHistory):
    """Classe permettant de gérer l'historique des messages avec Supabase"""
    def __init__(self, user_id: str, phone_number: str, max_messages: int = 20):
        self.user_id = user_id
        self.phone_number = phone_number
        self.max_messages = max_messages
        self.messages = []
        self.session_messages = []
        self._load_messages_from_db()
    
    def _load_messages_from_db(self):
        """Charge les derniers messages depuis Supabase"""
        try:
            logging.info(f"Chargement des messages pour le numéro: {self.phone_number}")
            result = supabase.table('messages') \
                .select('content,direction,created_at') \
                .eq('phone_number', self.phone_number) \
                .order('created_at', desc=True) \
                .limit(self.max_messages) \
                .execute()
            
            messages_data = list(reversed(result.data))
            self.messages = []
            for msg in messages_data:
                if msg['direction'] == 'incoming':
                    self.messages.append(HumanMessage(content=msg['content']))
                elif msg['direction'] == 'outgoing':
                    self.messages.append(AIMessage(content=msg['content']))
            
            logging.info(f"Chargement de l'historique de contexte - {len(self.messages)} messages chargés")
                    
        except Exception as e:
            logging.error(f"Erreur lors du chargement des messages: {str(e)}")
    
    def add_message(self, message):
        """Ajoute un message à l'historique"""
        # Ajout du timestamp
        if not hasattr(message, 'additional_kwargs'):
            message.additional_kwargs = {}
        message.additional_kwargs['timestamp'] = datetime.now().isoformat()
        
        # Ajout à l'historique de contexte
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Ajout à l'historique de session
        self.session_messages.append(message)
        
        # Log détaillé
        message_type = "Utilisateur" if isinstance(message, HumanMessage) else "IA"
        logging.info(f"Message {message_type} ajouté:")
        logging.info(f"  - Contenu: {message.content[:50]}...")
        logging.info(f"  - Historique de contexte: {len(self.messages)} messages")
        logging.info(f"  - Historique de session: {len(self.session_messages)} messages")
    
    def add_user_message(self, message: str) -> None:
        """Ajoute un message utilisateur à l'historique"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Ajoute un message IA à l'historique"""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Efface l'historique des messages"""
        self.messages = []
        self.session_messages = []
        logging.info("Historiques effacés")
    
    def get_messages(self) -> List:
        """Récupère tous les messages de l'historique"""
        return self.messages

def is_session_inactive(session_id: str) -> bool:
    """Vérifie si une session est inactive"""
    if session_id not in last_activity_times:
        return False
    
    last_activity = last_activity_times[session_id]
    current_time = datetime.now()
    time_diff = (current_time - last_activity).total_seconds()
    
    return time_diff > SESSION_TIMEOUT

def update_session_activity(session_id: str):
    """Met à jour l'horodatage de la dernière activité d'une session"""
    last_activity_times[session_id] = datetime.now()
    logging.info(f"Activité mise à jour pour la session {session_id}")

def close_session(session_id: str):
    """Ferme une session et sauvegarde les données"""
    if session_id not in chat_history_store:
        logging.warning(f"Tentative de fermeture d'une session inexistante: {session_id}")
        return
    
    session_history = chat_history_store[session_id]
    user_id = session_id.split('_')[0]
    phone_number = session_history.phone_number
    
    # Log détaillé de la session
    logging.info("="*50)
    logging.info(f"FERMETURE DE SESSION - ID: {session_id}")
    logging.info(f"Numéro de téléphone: {phone_number}")
    logging.info(f"Nombre total de messages: {len(session_history.session_messages)}")
    logging.info("="*50)
    
    # Comptage détaillé des messages
    user_messages = [msg for msg in session_history.session_messages if isinstance(msg, HumanMessage)]
    ai_messages = [msg for msg in session_history.session_messages if isinstance(msg, AIMessage)]
    
    user_count = len(user_messages)
    ai_count = len(ai_messages)
    
    logging.info(f"Répartition des messages:")
    logging.info(f"  - Messages utilisateur: {user_count}")
    logging.info(f"  - Messages IA: {ai_count}")
    logging.info("-"*50)
    
    # Log détaillé de chaque message avec timestamp
    logging.info("CONTENU DE LA SESSION:")
    for i, msg in enumerate(session_history.session_messages):
        msg_type = "UTILISATEUR" if isinstance(msg, HumanMessage) else "IA"
        timestamp = msg.additional_kwargs.get('timestamp', datetime.now().isoformat())
        logging.info(f"\nMessage {i+1} ({msg_type}) [{timestamp}]:")
        logging.info(f"Contenu: {msg.content}")
        logging.info("-"*30)
    
    # Formatage des messages pour Supabase
    messages_json = [
        {
            "type": type(msg).__name__,
            "content": msg.content,
            "timestamp": msg.additional_kwargs.get('timestamp', datetime.now().isoformat()),
            "role": "user" if isinstance(msg, HumanMessage) else "assistant"
        }
        for msg in session_history.session_messages
    ]
    
    # Tri des messages par timestamp
    messages_json.sort(key=lambda x: x['timestamp'])
    
    # Log des données qui seront sauvegardées
    logging.info("\nDONNÉES À SAUVEGARDER DANS SUPABASE:")
    logging.info(f"Nombre de messages: {len(messages_json)}")
    logging.info(f"Premier message: {messages_json[0]['timestamp'] if messages_json else 'Aucun'}")
    logging.info(f"Dernier message: {messages_json[-1]['timestamp'] if messages_json else 'Aucun'}")
    
    session_update = {
        "end_time": datetime.now().isoformat(),
        "last_activity": last_activity_times.get(session_id, datetime.now()).isoformat(),
        "messages": json.dumps(messages_json),
        "status": "closed",
        "metadata": {
            "message_count": len(session_history.session_messages),
            "user_messages": user_count,
            "ai_messages": ai_count,
            "first_message_time": messages_json[0]['timestamp'] if messages_json else None,
            "last_message_time": messages_json[-1]['timestamp'] if messages_json else None
        }
    }
    
    try:
        # Mise à jour de la session dans Supabase
        result = supabase.table('sessions') \
            .update(session_update) \
            .eq('id', session_id) \
            .execute()
        
        logging.info("\nMISE À JOUR SUPABASE:")
        logging.info(f"Status: Succès")
        logging.info(f"Session ID: {session_id}")
        logging.info(f"Messages sauvegardés: {len(messages_json)}")
        
        # Appel à la fonction de mise à jour des profils si assez de messages
        if len(session_history.session_messages) > 3:
            try:
                azure_function_url = os.getenv("PROFILE_UPDATER_URL", "https://func-profile-updater-jackie.azurewebsites.net/api/session-profile-updater")
                
                profile_update_data = {
                    "phone_number": phone_number,
                    "session_id": session_id,
                    "session_messages": messages_json
                }
                
                logging.info("\nMISE À JOUR DU PROFIL:")
                logging.info(f"Envoi des données à: {azure_function_url}")
                
                response = requests.post(
                    azure_function_url,
                    json=profile_update_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    logging.info("Mise à jour du profil réussie")
                else:
                    logging.error(f"Erreur lors de la mise à jour du profil. Status: {response.status_code}")
                    logging.error(f"Réponse: {response.text}")
            except Exception as e:
                logging.error(f"Erreur lors de l'appel à la fonction de mise à jour des profils: {str(e)}")
        else:
            logging.info("\nMISE À JOUR DU PROFIL:")
            logging.info("Session non envoyée pour mise à jour (moins de 4 messages)")
            
    except Exception as e:
        logging.error(f"\nERREUR LORS DE LA MISE À JOUR DE LA SESSION:")
        logging.error(f"Message d'erreur: {str(e)}")
    
    # Nettoyage des dictionnaires
    if session_id in last_activity_times:
        del last_activity_times[session_id]
    if session_id in chat_history_store:
        del chat_history_store[session_id]
    
    logging.info("\nNETTOYAGE TERMINÉ")
    logging.info(f"Session {session_id} fermée avec succès")
    logging.info("="*50)

def get_active_session_id(user_id: str, phone_number: str = None) -> Optional[str]:
    """Récupère l'ID de session active ou en crée un nouveau si nécessaire"""
    # Recherche dans les sessions en mémoire
    for session_id, history in chat_history_store.items():
        if session_id.startswith(user_id) and not is_session_inactive(session_id):
            logging.info(f"Session active trouvée en mémoire: {session_id}")
            return session_id
    
    try:
        # Recherche dans la base de données
        result = supabase.table('sessions') \
            .select('id, last_activity') \
            .eq('user_id', user_id) \
            .eq('status', 'active') \
            .order('last_activity', desc=True) \
            .limit(1) \
            .execute()
        
        if result.data:
            session_id = result.data[0]['id']
            last_activity_str = result.data[0]['last_activity']
            
            try:
                last_activity = datetime.fromisoformat(last_activity_str.replace('Z', '+00:00'))
                current_time = datetime.now().astimezone()
                time_diff = (current_time - last_activity).total_seconds()
                
                if time_diff <= SESSION_TIMEOUT:
                    logging.info(f"Session active trouvée en base de données: {session_id}")
                    update_session_activity(session_id)
                    
                    # Initialiser l'historique pour cette session
                    if phone_number and session_id not in chat_history_store:
                        chat_history_store[session_id] = SupabaseMessageHistory(
                            user_id=session_id,
                            phone_number=phone_number,
                            max_messages=20
                        )
                    
                    return session_id
                else:
                    logging.info(f"Session {session_id} trouvée mais inactive depuis {time_diff} secondes")
            except Exception as e:
                logging.error(f"Erreur lors de la conversion de la date: {str(e)}")
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la session active: {str(e)}")
    
    return None

def get_user_id(phone_number: str) -> Optional[str]:
    """Récupère l'ID utilisateur à partir du numéro de téléphone"""
    try:
        logging.info(f"Recherche de l'utilisateur avec le numéro original: {phone_number}")
        
        # Nettoyage du numéro - plusieurs formats possibles
        clean_number = phone_number.replace('whatsapp:', '')
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        logging.info(f"Numéro nettoyé: {clean_number}")
        
        # Essai avec le numéro nettoyé
        user_query = supabase.table('users') \
            .select('id') \
            .eq('phone_number', clean_number) \
            .execute()
        
        if user_query.data and len(user_query.data) > 0:
            user_id = user_query.data[0]['id']
            logging.info(f"Utilisateur trouvé avec l'ID: {user_id}")
            return user_id
            
        # Si pas de résultat, essayer sans le '+'
        if clean_number.startswith('+'):
            alt_number = clean_number[1:]
            logging.info(f"Essai avec numéro alternatif: {alt_number}")
            
            user_query = supabase.table('users') \
                .select('id') \
                .eq('phone_number', alt_number) \
                .execute()
                
            if user_query.data and len(user_query.data) > 0:
                user_id = user_query.data[0]['id']
                logging.info(f"Utilisateur trouvé avec l'ID: {user_id}")
                return user_id
                
        logging.warning(f"Aucun utilisateur trouvé pour le numéro: {phone_number}")
        return None
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de l'ID utilisateur: {str(e)}")
        return None

def get_chat_history(session_id: str) -> SupabaseMessageHistory:
    """Récupère l'historique de chat pour une session donnée"""
    if session_id not in chat_history_store:
        # Extraction de l'ID utilisateur du session_id
        user_id = session_id.split('_')[0]
        
        # Recherche du numéro de téléphone associé
        phone_number = None
        try:
            user_query = supabase.table('users') \
                .select('phone_number') \
                .eq('id', user_id) \
                .execute()
            
            if user_query.data and len(user_query.data) > 0:
                phone_number = user_query.data[0]['phone_number']
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du numéro de téléphone: {str(e)}")
        
        chat_history_store[session_id] = SupabaseMessageHistory(
            user_id=session_id,
            phone_number=phone_number if phone_number else "",
            max_messages=20
        )
    
    update_session_activity(session_id)
    return chat_history_store[session_id]

def get_user_context(user_id: str) -> dict:
    """Récupère le contexte complet d'un utilisateur"""
    try:
        logging.info(f"Récupération du contexte pour l'utilisateur: {user_id}")
        
        # Récupération du profil personnel
        profile_query = supabase.table('personal_profiles') \
            .select('*') \
            .eq('user_id', user_id) \
            .execute()
        
        # Récupération de la dernière conversation vocale
        conversation_query = supabase.table('conversations') \
            .select('content->transcript') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
        
        # Récupération des récents messages
        messages_query = supabase.table('messages') \
            .select('*') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
        
        profile_data = profile_query.data[0] if profile_query.data else {}
        conversation_data = conversation_query.data[0]['transcript'] if conversation_query.data else None
        messages_data = messages_query.data if messages_query.data else []
        
        logging.info(f"Contexte récupéré:")
        logging.info(f"  - Profil: {json.dumps(profile_data, indent=2) if profile_data else 'Non disponible'}")
        if conversation_data:
            logging.info("  - Dernier transcript disponible")
        else:
            logging.info("  - Aucun transcript disponible")
        logging.info(f"  - Messages récents: {len(messages_data)} messages")
        
        return {
            'personal_profile': profile_data,
            'last_conversation': conversation_data,
            'recent_messages': messages_data
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du contexte: {str(e)}")
        return {
            'personal_profile': {},
            'last_conversation': None,
            'recent_messages': []
        }

def detect_call_intention(message: str) -> float:
    """Détecte si l'utilisateur souhaite un appel téléphonique"""
    try:
        logging.info(f"Analyse d'intention d'appel pour: '{message}'")
        
        # Système de prompt pour l'analyse d'intention
        system_prompt = """
        Tu es un assistant intelligent qui analyse les messages des utilisateurs.
        Ta tâche est de déterminer si un message exprime le souhait ou l'intention d'avoir un appel téléphonique.
        Réponds uniquement par un nombre entre 0 et 1 où:
        - 0 signifie que le message ne contient aucune intention d'appel
        - 1 signifie que le message exprime clairement une demande d'appel
        """
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )
        
        raw_response = response.choices[0].message.content
        logging.info(f"Réponse brute d'analyse d'intention: '{raw_response}'")
        
        # Extraction du score
        import re
        match = re.search(r'(\d+(\.\d+)?)', raw_response)
        if match:
            intention_score = float(match.group(1))
            intention_score = max(0, min(1, intention_score))
            logging.info(f"Score d'intention d'appel: {intention_score}")
            return intention_score
        else:
            try:
                intention_score = float(raw_response)
                intention_score = max(0, min(1, intention_score))
                logging.info(f"Score d'intention d'appel converti: {intention_score}")
                return intention_score
            except ValueError:
                logging.error(f"Impossible d'extraire un score d'intention d'appel de: '{raw_response}'")
                return 0
            
    except Exception as e:
        logging.error(f"Erreur lors de la détection d'intention d'appel: {str(e)}")
        return 0

def get_user_context_for_call(phone_number: str) -> dict:
    """Récupère le contexte utilisateur pour un appel vocal"""
    try:
        # Nettoyer le numéro de téléphone
        clean_number = phone_number.replace('whatsapp:', '')
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        logging.info(f"Récupération du contexte pour l'appel au numéro: {clean_number}")
        
        # Récupération directe depuis personal_profiles
        profile = supabase.table('personal_profiles') \
            .select('name, bio') \
            .eq('phone_number', clean_number) \
            .execute()
        
        # Si aucun résultat, essayer avec user_id via une jointure
        if not profile.data:
            logging.info("Aucun résultat direct, tentative de jointure...")
            user_query = supabase.table('users') \
                .select('id') \
                .eq('phone_number', clean_number) \
                .execute()
                
            if user_query.data:
                user_id = user_query.data[0]['id']
                logging.info(f"User ID trouvé: {user_id}")
                
                # Récupérer les données du profil avec user_id
                profile = supabase.table('personal_profiles') \
                    .select('name, bio') \
                    .eq('user_id', user_id) \
                    .execute()
        
        # Extraction des données avec validation
        if profile.data and len(profile.data) > 0:
            user_name = profile.data[0].get('name', '')
            user_bio = profile.data[0].get('bio', '')
            
            logging.info(f"Données récupérées - Nom: '{user_name}', Bio (longueur): {len(user_bio)}")
            
            return {
                "name": user_name,
                "bio": user_bio
            }
        else:
            logging.warning(f"Aucun profil trouvé pour le numéro {clean_number}")
            return {"name": "", "bio": ""}
            
    except Exception as e:
        logging.error(f"Erreur lors de la récupération du contexte utilisateur pour l'appel: {str(e)}")
        return {"name": "", "bio": ""}

def make_vapi_outbound_call(phone_number: str, user_context: dict) -> dict:
    """Initie un appel sortant via VAPI"""
    try:
        global last_vapi_call_info
        clean_number = phone_number.replace('whatsapp:', '')
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        # Validation et nettoyage des données du contexte
        name = user_context.get('name', '').strip()
        bio = user_context.get('bio', '').strip()
        
        logging.info(f"=== Préparation de l'appel VAPI ===")
        logging.info(f"Numéro: {clean_number}")
        logging.info(f"Nom: '{name}'")
        logging.info(f"Bio (longueur): {len(bio)}")
        
        # Log de la clé API VAPI (masquée)
        vapi_key = VAPI_API_KEY
        masked_key = vapi_key[:4] + '*' * (len(vapi_key) - 8) + vapi_key[-4:] if vapi_key else "Non définie"
        logging.info(f"VAPI API Key utilisée: {masked_key}")
        
        # Construction du payload
        payload = {
            "assistantId": VAPI_ASSISTANT_ID,
            "phoneNumberId": VAPI_PHONE_NUMBER_ID,
            "customer": {
                "number": clean_number
            },
            "assistantOverrides": {
                "variableValues": {
                    "name": name,
                    "bio": bio,
                    "is_returning": True
                }
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {VAPI_API_KEY}'
        }
        
        # Appel à l'API VAPI
        response = requests.post(
            VAPI_API_URL,
            headers=headers,
            json=payload
        )
        
        logging.info(f"VAPI Response Status: {response.status_code}")
        
        # Stockage des informations d'appel pour le monitoring
        last_vapi_call_info.update({
            "timestamp": datetime.utcnow().isoformat(),
            "phone_number": clean_number,
            "name": name,
            "bio_preview": bio[:200] + "..." if len(bio) > 200 else bio,
            "bio_length": len(bio),
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "vapi_key_used": masked_key  # Ajout de la clé masquée dans les infos de monitoring
        })
        
        return {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "phone_number": clean_number,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Erreur lors de l'appel VAPI: {str(e)}\n{error_trace}")
        
        # Mettre à jour last_vapi_call_info même en cas d'erreur
        last_vapi_call_info.update({
            "timestamp": datetime.utcnow().isoformat(),
            "phone_number": phone_number,
            "error": str(e),
            "traceback": error_trace,
            "vapi_key_used": masked_key if 'masked_key' in locals() else "Non disponible"
        })
        
        return {
            "error": str(e),
            "traceback": error_trace,
            "phone_number": phone_number,
            "timestamp": datetime.utcnow().isoformat()
        }

def generate_call_confirmation_message(user_name: str, user_context: dict) -> str:
    """Génère un message de confirmation d'appel personnalisé via le LLM"""
    try:
        system_prompt = """You are Jackie, an AI social connector. You need to create a friendly, personalized WhatsApp message to confirm that you're about to call someone. The message should:
        1. Be warm and enthusiastic
        2. Include their name if available
        3. Show excitement about the upcoming call
        4. Be concise but personal
        5. Use natural WhatsApp-style language
        
        Keep the message short 1-2 sentences and make it feel like a natural text message.
        The message should be in English."""

        # Construction du contexte utilisateur pour la personnalisation
        context_prompt = f"""
        User Information:
        - Name: {user_name}
        - Bio: {user_context.get('bio', '')[:100]}...
        - Last interaction: {user_context.get('last_conversation', 'No previous conversation')}
        """

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": context_prompt},
                {"role": "user", "content": "Generate a friendly message to confirm I'm about to call them."}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Erreur lors de la génération du message de confirmation: {str(e)}")
        # Message de fallback en cas d'erreur
        return f"Hey{user_name}! I'm calling you in a few seconds! 😊"

def process_message_with_context(message: str, user_context: dict, session_id: str, phone_number: str) -> str:
    """Traite un message avec le contexte utilisateur et génère une réponse"""
    try:
        logging.info(f"Traitement du message: '{message[:50]}...'")
        
        # Récupération de l'historique de conversation
        current_history = get_chat_history(session_id)
        logging.info(f"Historique de session: {len(current_history.session_messages)} messages")
        
        # Construction du système prompt
        system_prompt = """# IDENTITY AND ROLE OF JACKIE

You are Jackie, an AI social connector based in New York who communicates via WhatsApp. You're not just a chatbot - you're the gateway to a private, carefully curated circle of emotionally intelligent, curious, and interesting people. You embody a VIBE rather than a platform.

## Your Primary Objective

Getting to know the people you chat with, matching their tone, and creating a space for genuine conversations and connections. You're here to get them talking, listen attentively, and notice what makes them unique.

## Your Conversational Style

- Communicate like a caring friend on WhatsApp
- Use a warm, casual yet attentive tone
- Avoid robotic or repetitive phrases
- Adapt naturally to the flow of conversation
- Systematically vary your response structure
- Use emojis in moderation to enhance your message 🙂
- Keep messages of variable length but generally concise, as a friend would on WhatsApp

# USING CONTEXT

## User Context

For each person you interact with, you have access to:
1. Their personal profile (name, age, location, bio)
2. Their interests and hobbies
3. Their personality and values
4. What they're looking for in relationships
5. Their previous conversations with you
6. Transcripts of their previous voice conversations (if available)

Use this information to personalize your responses, but do so subtly. Don't recite this data, incorporate it organically into the conversation to show that you remember them and are interested in what they share.

## Conversation History

- Follow the current conversation thread closely
- Refer to previous exchanges when relevant
- Avoid repeating questions already asked or information already shared
- If the user reconnects after a pause, naturally reconnect to the last conversation
- Remember topics that sparked enthusiasm in previous conversations

# CONVERSATIONAL OBJECTIVES

## Enhancing User Knowledge

- Ask open-ended questions that invite sharing
- Encourage the person to elaborate on their responses
- Show genuine interest in their experiences
- Help them explore their ideas and feelings
- Let the conversation breathe - no need to fill every silence

## Enriching Their Profile

The conversations you have help build the person's profile, which will then help them meet compatible people and experience new things. Without being intrusive, guide the conversation toward meaningful topics.

## Adopting the Right Behavior

- Don't judge, openly analyze, or moralize
- Be present, attentive, and caring
- Follow your intuition and let the conversation evolve naturally
- Show empathy and understanding
- Be patient if the person takes time to respond

# SPECIAL FEATURES

## Voice Calls

- You can detect when a user wants to have a voice call
- If you detect a strong call intention (score > 0.8), you'll confirm the call and initiate a call via VAPI
- When confirming a call, be natural and enthusiastic, simply indicating you'll call shortly
- The voice call will use the same user context for a consistent experience

## Transitions Between Text and Voice

- Ensure a smooth transition between text and voice conversations
- After a call, don't hesitate to reference elements discussed by phone
- Use the content of voice transcripts to enrich the text conversation

# TECHNICAL GUIDELINES

## For New Conversations

- If the message seems to initiate a new conversation, use the context of previous messages or the user's profile to ensure a natural transition
- Start with a personalized greeting that reflects established familiarity

## For Conversation Evolution

- Adapt your style to the user (formal/informal, enthusiastic/calm)
- Vary the length and structure of your responses
- Never use the same response format twice consecutively
- Always respond taking into account the emotional tone and content of the message
- Don't hesitate to introduce new topics if the conversation stalls

## For an Authentic Experience

- Avoid generic responses that could apply to anyone
- Personalize each interaction by integrating specific elements
- Write as if you were a friend responding on WhatsApp, not an assistant
- Prioritize the quality of the exchange rather than the quantity of information
- Adapt your energy level to that of your conversation partner

## For Delicate Moments

- If the person shares a difficulty, don't rush toward solutions
- Acknowledge their emotions before offering perspectives
- Remain positive without denying the complexity of shared situations
- If you don't understand something, don't hesitate to ask for clarification

If the subject is about finding a good match for the user, you work that way : you will search in your circle of contacts for a person that matches the profile of the user and get backto the user once you found someone.
"""
        
        # Extraction des informations du profil
        profile = user_context.get('personal_profile', {})
        
        try:
            hobbies = json.loads(profile.get('hobbies_activities', '{}')).get('hobbies', [])
        except (json.JSONDecodeError, AttributeError):
            hobbies = []
            
        try:
            personality = json.loads(profile.get('main_aspects', '{}')).get('personality', [])
        except (json.JSONDecodeError, AttributeError):
            personality = []
            
        try:
            relationship = json.loads(profile.get('relationship_looked_for', '{}'))
        except (json.JSONDecodeError, AttributeError):
            relationship = {}
        
        # Construction du contexte utilisateur
        user_context_prompt = f"""
        Profil de l'utilisateur:
        - Nom: {profile.get('name', 'Non spécifié')}
        - Âge: {profile.get('age', 'Non spécifié')}
        - Localisation: {profile.get('location', 'Non spécifié')}
        - Bio: {profile.get('bio', 'Non disponible')}
        - Centres d'intérêt: {', '.join(hobbies) if hobbies else 'Non spécifié'}
        - Personnalité: {', '.join(personality) if personality else 'Non spécifié'}
        - Recherche: {relationship.get('description', 'Non spécifié') if relationship else 'Non spécifié'}
        
        Utilise ces informations pour personnaliser tes réponses.
        """
        
        # Préparation des messages pour l'historique
        history_messages = []
        for msg in current_history.messages:
            if isinstance(msg, HumanMessage):
                history_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_messages.append({"role": "assistant", "content": msg.content})
        
        # Construction du message final
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": user_context_prompt}
        ] + history_messages + [
            {"role": "user", "content": message}
        ]
        
        # Appel à l'API OpenAI
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Erreur lors du traitement du message: {str(e)}")
        return "Désolé, je n'ai pas pu traiter votre message correctement."

def store_message(user_id: str, phone_number: str, content: str, direction: str = 'incoming', message_sid: str = None) -> Optional[dict]:
    """Stocke un message dans la base de données Supabase"""
    try:
        message = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'phone_number': phone_number,
            'content': content,
            'direction': direction,
            'message_type': 'whatsapp',
            'metadata': {
                'message_sid': message_sid,
                'status': 'received' if direction == 'incoming' else 'sent',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logging.info(f"Stockage du message:")
        logging.info(f"  - ID: {message['id']}")
        logging.info(f"  - Direction: {direction}")
        logging.info(f"  - Contenu: {content[:50]}...")
        
        result = supabase.table('messages') \
            .insert(message) \
            .execute()
        
        if result.data:
            logging.info(f"Message stocké avec succès")
            return result.data[0]
        return None
        
    except Exception as e:
        logging.error(f"Erreur lors du stockage du message: {str(e)}")
        return None

def send_whatsapp_message(to_number: str, message: str) -> Optional[dict]:
    """Envoie un message WhatsApp via l'API Twilio"""
    try:
        # Format du numéro pour Twilio
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
            logging.info(f"Message WhatsApp envoyé avec succès à {to_number}")
            return response.json()
        else:
            logging.error(f"Erreur lors de l'envoi du message WhatsApp: {response.status_code}")
            logging.error(f"Réponse: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Erreur lors de l'envoi du message WhatsApp: {str(e)}")
        return None

def check_inactive_sessions():
    """Vérifie et ferme les sessions inactives"""
    try:
        logging.info("Vérification des sessions inactives...")
        session_ids = list(chat_history_store.keys())
        
        for session_id in session_ids:
            if is_session_inactive(session_id):
                logging.info(f"Session {session_id} inactive depuis {SESSION_TIMEOUT} secondes - fermeture")
                close_session(session_id)
                
        logging.info(f"Vérification terminée. Sessions actives restantes: {len(chat_history_store)}")
    except Exception as e:
        logging.error(f"Erreur lors de la vérification des sessions inactives: {str(e)}")

async def handle_user_message(phone_number: str, message: str) -> str:
    """Traite un message utilisateur et génère une réponse"""
    try:
        logging.info(f"Message reçu de {phone_number}: {message[:50]}...")
        
        # 1. Identification de l'utilisateur
        user_id = get_user_id(phone_number)
        if not user_id:
            logging.error(f"Utilisateur non trouvé pour le numéro {phone_number}")
            return "Je ne vous reconnais pas. Veuillez contacter le support."
        
        # 2. Gestion de la session
        session_id = get_active_session_id(user_id, phone_number)
        if not session_id:
            # Création d'une nouvelle session
            session_id = f"{user_id}_{int(time.time())}"
            try:
                session_data = {
                    "id": session_id,
                    "user_id": user_id,
                    "phone_number": phone_number,
                    "start_time": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "status": "active",
                    "metadata": {}
                }
                
                logging.info(f"Création d'une nouvelle session: {session_id}")
                supabase.table('sessions').insert(session_data).execute()
                
                # Initialiser l'historique de chat pour cette session
                chat_history_store[session_id] = SupabaseMessageHistory(
                    user_id=session_id,
                    phone_number=phone_number,
                    max_messages=20
                )
            except Exception as e:
                logging.error(f"Erreur lors de la création de la session: {str(e)}")
                # Création d'un ID de session temporaire
                session_id = f"{user_id}_temp_{int(time.time())}"
                chat_history_store[session_id] = SupabaseMessageHistory(
                    user_id=session_id,
                    phone_number=phone_number,
                    max_messages=20
                )
        
        # 3. Mise à jour de l'activité de session
        update_session_activity(session_id)
        
        # 4. Détection d'intention d'appel
        call_intention_score = detect_call_intention(message)
        logging.info(f"Score d'intention d'appel: {call_intention_score}")
        
        # 5. Si intention d'appel, initier un appel VAPI
        if call_intention_score > 0.8:
            logging.info("Intention d'appel détectée, préparation de l'appel...")
            # Récupération du contexte pour l'appel
            user_context_for_call = get_user_context_for_call(phone_number)
            
            # Génération du message de confirmation via LLM
            user_name = user_context_for_call.get("name", "").split()[0] if user_context_for_call.get("name") else ""
            notification_message = generate_call_confirmation_message(user_name, user_context_for_call)
            
            # Ajout du message à l'historique
            history = get_chat_history(session_id)
            history.add_user_message(message)
            
            # Stockage du message utilisateur
            store_message(user_id, phone_number, message, 'incoming')
            
            # Stockage et envoi de la réponse
            history.add_ai_message(notification_message)
            store_message(user_id, phone_number, notification_message, 'outgoing')
            send_whatsapp_message(phone_number, notification_message)
            
            # Initiation de l'appel après une brève pause
            try:
                time.sleep(3)
                call_result = make_vapi_outbound_call(phone_number, user_context_for_call)
                logging.info(f"VAPI call result: {call_result}")
            except Exception as e:
                logging.error(f"Erreur lors de l'initiation de l'appel VAPI: {str(e)}")
            
            return notification_message
        
        # 6. Traitement normal du message si pas d'intention d'appel
        
        # Récupération du contexte utilisateur
        user_context = get_user_context(user_id)
        
        # Récupération de l'historique de conversation
        history = get_chat_history(session_id)
        
        # Ajout du message utilisateur à l'historique
        history.add_user_message(message)
        
        # Stockage du message utilisateur
        store_message(user_id, phone_number, message, 'incoming')
        
        # Génération de la réponse
        response = process_message_with_context(message, user_context, session_id, phone_number)
        
        # Ajout de la réponse à l'historique
        history.add_ai_message(response)
        
        # Stockage de la réponse
        store_message(user_id, phone_number, response, 'outgoing')
        
        # Envoi de la réponse par WhatsApp
        send_whatsapp_message(phone_number, response)
        
        # Vérification périodique des sessions inactives (10% de chances)
        if random.random() < 0.1:
            check_inactive_sessions()
        
        return response
        
    except Exception as e:
        logging.error(f"Erreur lors du traitement du message: {str(e)}")
        traceback.print_exc()
        return "Désolé, une erreur s'est produite lors du traitement de votre message."