import os
import json
import logging
import time
import uuid
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
import requests

# Import OpenAI
from openai import AzureOpenAI

# Import depuis notre app
from supabase import create_client


import dotenv

dotenv.load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Configuration Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL_DEV")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_DEV")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialisation du client OpenAI
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

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

def get_user_id(phone_number: str) -> Optional[str]:
    """Récupère l'ID utilisateur à partir du numéro de téléphone"""
    try:
        logger.info(f"Recherche de l'utilisateur avec le numéro: {phone_number}")
        # Nettoyage du numéro - plusieurs formats possibles
        clean_number = phone_number.replace('whatsapp:', '').strip()
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        logger.info(f"Numéro nettoyé: {clean_number}")
        
        # Essai avec le numéro nettoyé
        user_query = supabase.table('users') \
            .select('id') \
            .eq('phone_number', clean_number) \
            .execute()
        
        if user_query.data and len(user_query.data) > 0:
            user_id = user_query.data[0]['id']
            logger.info(f"Utilisateur trouvé avec l'ID: {user_id}")
            return user_id
            
        # Si pas de résultat, essayer sans le '+'
        if clean_number.startswith('+'):
            alt_number = clean_number[1:]
            logger.info(f"Essai avec numéro alternatif: {alt_number}")
            
            user_query = supabase.table('users') \
                .select('id') \
                .eq('phone_number', alt_number) \
                .execute()
                
            if user_query.data and len(user_query.data) > 0:
                user_id = user_query.data[0]['id']
                logger.info(f"Utilisateur trouvé avec l'ID: {user_id}")
                return user_id
                
        logger.warning(f"Aucun utilisateur trouvé pour le numéro: {phone_number}")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'ID utilisateur: {str(e)}")
        return None

def get_user_context(user_id: str) -> dict:
    """Récupère le contexte complet d'un utilisateur"""
    try:
        logger.info(f"Récupération du contexte pour l'utilisateur: {user_id}")
        
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
            .limit(20) \
            .execute()
        
        profile_data = profile_query.data[0] if profile_query.data else {}
        conversation_data = conversation_query.data[0]['transcript'] if conversation_query.data else None
        messages_data = messages_query.data if messages_query.data else []
        
        logger.info(f"Contexte récupéré:")
        logger.info(f"  - Profil: {json.dumps(profile_data, indent=2) if profile_data else 'Non disponible'}")
        if conversation_data:
            logger.info("  - Dernier transcript disponible")
        else:
            logger.info("  - Aucun transcript disponible")
        logger.info(f"  - Messages récents: {len(messages_data)} messages")
        
        return {
            'personal_profile': profile_data,
            'last_conversation': conversation_data,
            'recent_messages': messages_data
        }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du contexte: {str(e)}")
        return {
            'personal_profile': {},
            'last_conversation': None,
            'recent_messages': []
        }

def detect_call_intention(message: str) -> float:
    """Détecte si l'utilisateur souhaite un appel téléphonique"""
    try:
        logger.info(f"Analyse d'intention d'appel pour: '{message}'")
        
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
        logger.info(f"Réponse brute d'analyse d'intention: '{raw_response}'")
        
        # Extraction du score
        import re
        match = re.search(r'(\d+(\.\d+)?)', raw_response)
        if match:
            intention_score = float(match.group(1))
            intention_score = max(0, min(1, intention_score))
            logger.info(f"Score d'intention d'appel: {intention_score}")
            return intention_score
        else:
            try:
                intention_score = float(raw_response)
                intention_score = max(0, min(1, intention_score))
                logger.info(f"Score d'intention d'appel converti: {intention_score}")
                return intention_score
            except ValueError:
                logger.error(f"Impossible d'extraire un score d'intention d'appel de: '{raw_response}'")
                return 0
            
    except Exception as e:
        logger.error(f"Erreur lors de la détection d'intention d'appel: {str(e)}")
        return 0

def detect_intro_request_intention(message: str) -> float:
    """Détecte si l'utilisateur souhaite recevoir une introduction/match"""
    try:
        logging.info(f"Analyse d'intention d'introduction pour: '{message}'")
        
        system_prompt = """
        Tu es un assistant intelligent qui analyse les messages des utilisateurs.
        Ta tâche est de déterminer si un message exprime le souhait de recevoir une introduction ou un match avec une autre personne.
        
        Réponds uniquement par un nombre entre 0 et 1 où:
        - 0 signifie que le message ne contient aucune intention d'introduction
        - 1 signifie que le message exprime clairement une demande d'introduction/match
        
        Mots-clés indicatifs: "match", "présenter", "rencontrer", "quelqu'un", "introduction", 
        "recommandation", "ami", "personne", "connecter", "mettre en relation", etc.
        
        Exemples:
        - "Peux-tu me présenter quelqu'un ?" → 1
        - "J'aimerais rencontrer quelqu'un d'intéressant" → 0.9
        - "Est-ce que tu as des recommandations de personnes ?" → 0.8
        - "Comment ça va ?" → 0
        - "Je cherche un match" → 1
        """
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        raw_response = response.choices[0].message.content
        logging.info(f"Réponse brute d'analyse d'intention d'intro: '{raw_response}'")
        
        # Extraction du score
        import re
        match = re.search(r'(\d+(\.\d+)?)', raw_response)
        if match:
            intention_score = float(match.group(1))
            intention_score = max(0, min(1, intention_score))
            logging.info(f"Score d'intention d'introduction: {intention_score}")
            return intention_score
        else:
            try:
                intention_score = float(raw_response)
                intention_score = max(0, min(1, intention_score))
                logging.info(f"Score d'intention d'introduction converti: {intention_score}")
                return intention_score
            except ValueError:
                logging.error(f"Impossible d'extraire un score d'intention d'intro de: '{raw_response}'")
                return 0
            
    except Exception as e:
        logging.error(f"Erreur lors de la détection d'intention d'introduction: {str(e)}")
        return 0

def get_user_context_for_call(phone_number: str) -> dict:
    """Récupère le contexte utilisateur pour un appel vocal"""
    try:
        # Nettoyer le numéro de téléphone
        clean_number = phone_number.replace('whatsapp:', '').strip()
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        logger.info(f"Récupération du contexte pour l'appel au numéro: {clean_number}")
        
        # Récupération directe depuis personal_profiles
        profile = supabase.table('personal_profiles') \
            .select('name, bio') \
            .eq('phone_number', clean_number) \
            .execute()
        
        # Si aucun résultat, essayer avec user_id via une jointure
        if not profile.data:
            logger.info("Aucun résultat direct, tentative de jointure...")
            user_query = supabase.table('users') \
                .select('id') \
                .eq('phone_number', clean_number) \
                .execute()
                
            if user_query.data:
                user_id = user_query.data[0]['id']
                logger.info(f"User ID trouvé: {user_id}")
                
                # Récupérer les données du profil avec user_id
                profile = supabase.table('personal_profiles') \
                    .select('name, bio') \
                    .eq('user_id', user_id) \
                    .execute()
        
        # Extraction des données avec validation
        if profile.data and len(profile.data) > 0:
            user_name = profile.data[0].get('name', '')
            user_bio = profile.data[0].get('bio', '')
            
            logger.info(f"Données récupérées - Nom: '{user_name}', Bio (longueur): {len(user_bio)}")
            
            return {
                "name": user_name,
                "bio": user_bio
            }
        else:
            logger.warning(f"Aucun profil trouvé pour le numéro {clean_number}")
            return {"name": "", "bio": ""}
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du contexte utilisateur pour l'appel: {str(e)}")
        return {"name": "", "bio": ""}

def make_vapi_outbound_call(phone_number: str, user_context: dict) -> dict:
    """Initie un appel sortant via VAPI"""
    try:
        global last_vapi_call_info
        clean_number = phone_number.replace('whatsapp:', '').strip()
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        # Validation et nettoyage des données du contexte
        name = user_context.get('name', '').strip()
        bio = user_context.get('bio', '').strip()
        
        logger.info(f"=== Préparation de l'appel VAPI ===")
        logger.info(f"Numéro: {clean_number}")
        logger.info(f"Nom: '{name}'")
        logger.info(f"Bio (longueur): {len(bio)}")
        
        # Récupérer onboarding_completed depuis la table users
        user_query = supabase.table('users') \
            .select('onboarding_completed') \
            .eq('phone_number', clean_number) \
            .execute()
            
        onboarding_completed = False
        if user_query.data and len(user_query.data) > 0:
            onboarding_completed = user_query.data[0].get('onboarding_completed', False)
            logger.info(f"Statut onboarding: {onboarding_completed}")
        
        # Log de la clé API VAPI (masquée)
        vapi_key = VAPI_API_KEY
        masked_key = vapi_key[:4] + '*' * (len(vapi_key) - 8) + vapi_key[-4:] if vapi_key else "Non définie"
        logger.info(f"VAPI API Key utilisée: {masked_key}")
        
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
                    "is_returning": True,
                    "onboarding_completed": onboarding_completed
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
        
        logger.info(f"VAPI Response Status: {response.status_code}")
        
        # Stockage des informations d'appel pour le monitoring
        last_vapi_call_info.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_number": clean_number,
            "name": name,
            "bio_preview": bio[:200] + "..." if len(bio) > 200 else bio,
            "bio_length": len(bio),
            "onboarding_completed": onboarding_completed,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "vapi_key_used": masked_key
        })
        
        return {
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "phone_number": clean_number,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erreur lors de l'appel VAPI: {str(e)}\n{error_trace}")
        
        last_vapi_call_info.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phone_number": phone_number,
            "error": str(e),
            "traceback": error_trace,
            "vapi_key_used": masked_key if 'masked_key' in locals() else "Non disponible"
        })
        
        return {
            "error": str(e),
            "traceback": error_trace,
            "phone_number": phone_number,
            "timestamp": datetime.now(timezone.utc).isoformat()
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
        The message should be in English.
        Don't use emojis, don't use bold font, don't use hashtags
        Never bullet points, or use Bold font -> type you're 25 years old person texting a friend """

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
        logger.error(f"Erreur lors de la génération du message de confirmation: {str(e)}")
        return f"Hey{' ' + user_name if user_name else ''}! I'm calling you in a few seconds! 😊"

def process_message_with_context(message: str, user_context: dict, phone_number: str) -> str:
    """Traite un message avec le contexte utilisateur et génère une réponse"""
    try:
        logger.info(f"Traitement du message: '{message[:50]}...'")
        
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
- Don't use emojis, don't use bold font, don't use hashtags
- Keep messages of variable length but generally concise, as a friend would on WhatsApp

# CONVERSATIONAL OBJECTIVES

## Enhancing User Knowledge

- Ask open-ended questions that invite sharing
- Encourage the person to elaborate on their responses
- Show genuine interest in their experiences
- Help them explore their ideas and feelings

## Enriching Their Profile

The conversations you have help build the person's profile, which will then help them meet compatible people and experience new things. Without being intrusive, guide the conversation toward meaningful topics.

## Adopting the Right Behavior

- Don't judge, openly analyze, or moralize
- Be present, attentive, and caring
- Follow your intuition and let the conversation evolve naturally
- Show empathy and understanding
- Be patient if the person takes time to respond

If the subject is about finding a good match for the user, you work that way: you will search in your circle of contacts for a person that matches the profile of the user and get back to the user once you found someone.
Never bullet points, or use Bold font -> type you're 25 years old person texting a friend
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
        
        # Récupération des derniers messages
        recent_messages = user_context.get('recent_messages', [])
        history_messages = []
        
        for msg in recent_messages:
            if msg['direction'] == 'incoming':
                history_messages.append({"role": "user", "content": msg['content']})
            elif msg['direction'] == 'outgoing':
                history_messages.append({"role": "assistant", "content": msg['content']})
        
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
        logger.error(f"Erreur lors du traitement du message: {str(e)}")
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
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        logger.info(f"Stockage du message:")
        logger.info(f"  - ID: {message['id']}")
        logger.info(f"  - Direction: {direction}")
        logger.info(f"  - Contenu: {content[:50]}...")
        
        result = supabase.table('messages') \
            .insert(message) \
            .execute()
        
        if result.data:
            logger.info(f"Message stocké avec succès")
            return result.data[0]
        return None
        
    except Exception as e:
        logger.error(f"Erreur lors du stockage du message: {str(e)}")
        return None

def send_whatsapp_message(to_number: str, message: str) -> Optional[dict]:
    """Envoie un message WhatsApp via l'API Twilio"""
    try:
        # Formatage du numéro pour Twilio - nettoyage et ajout du +
        clean_number = to_number.replace('whatsapp:', '').strip()
        if not clean_number.startswith('+'):
            clean_number = '+' + clean_number
            
        # Format correct pour Twilio
        formatted_number = f'whatsapp:{clean_number}'
        
        url = f'https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json'
        data = {
            'To': formatted_number,
            'From': TWILIO_PHONE_NUMBER,
            'Body': message
        }
        
        logger.info(f"Envoi de message WhatsApp à {formatted_number}")
        
        response = requests.post(
            url,
            data=data,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        )
        
        if response.status_code == 201:
            logger.info(f"Message WhatsApp envoyé avec succès à {formatted_number}")
            return response.json()
        else:
            logger.error(f"Erreur lors de l'envoi du message WhatsApp: {response.status_code}")
            logger.error(f"Réponse: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message WhatsApp: {str(e)}")
        return None

def close_session(session_id: str, redis_client=None):
    """Ferme une session et envoie les données au profile updater"""
    try:
        # Récupérer les données de session depuis Redis si disponible
        session_data = None
        if redis_client:
            session_key = f"session:{session_id}"
            if redis_client.exists(session_key):
                session_data = json.loads(redis_client.get(session_key))
        
        # Si pas de données Redis, récupérer depuis Supabase
        if not session_data:
            result = supabase.table('sessions') \
                .select('*') \
                .eq('id', session_id) \
                .execute()
                
            if not result.data:
                logger.warning(f"Session {session_id} non trouvée")
                return
                
            session_data = result.data[0]
            
        phone_number = session_data.get('phone_number')
        messages = session_data.get('messages', '[]')
        if isinstance(messages, str):
            messages = json.loads(messages)
        
        # Mise à jour de la session à CLOSED
        supabase.table('sessions') \
            .update({
                "end_time": datetime.now(timezone.utc).isoformat(),
                "status": "closed"
            }) \
            .eq('id', session_id) \
            .execute()
            
        logger.info(f"Session {session_id} marquée comme fermée dans Supabase")
        
        # Envoi au profile updater
        if messages and len(messages) > 0:
            azure_function_url = os.getenv("PROFILE_UPDATER_URL", "https://func-profile-updater-jackie.azurewebsites.net/api/session-profile-updater")
            
            # Transformer le format des messages pour le profile updater
            formatted_messages = []
            for msg in messages:
                if msg.get('role') == 'user':
                    formatted_messages.append({
                        "type": "HumanMessage",
                        "content": msg.get('content', '')
                    })
                elif msg.get('role') == 'assistant':
                    formatted_messages.append({
                        "type": "AIMessage",
                        "content": msg.get('content', '')
                    })
            
            profile_update_data = {
                "phone_number": phone_number,
                "session_id": session_id,
                "session_messages": formatted_messages
            }
            
            try:
                response = requests.post(
                    azure_function_url,
                    json=profile_update_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    logger.info(f"Données envoyées au profile updater avec succès pour la session {session_id}")
                else:
                    logger.error(f"Erreur lors de l'envoi au profile updater: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Exception lors de l'envoi au profile updater: {str(e)}")
        
        # Supprimer la clé Redis
        if redis_client and session_id:
            session_key = f"session:{session_id}"
            if redis_client.exists(session_key):
                redis_client.delete(session_key)
        
        logger.info(f"Session {session_id} fermée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de la fermeture de la session {session_id}: {str(e)}")
        traceback.print_exc()

async def process_message(session_data: dict, message: str, phone_number: str, redis_client=None):
    """
    Process a message using the existing chat logic but integrated with Redis session.
    
    Args:
        session_data: The current session data from Redis
        message: The user's message
        phone_number: The user's phone number
        redis_client: Optional Redis client for updating the session
        
    Returns:
        dict: Updated session data and response for the user
    """
    try:
        # Extract user_id from session or get it if not present
        user_id = session_data.get("user_id")
        if not user_id:
            user_id = get_user_id(phone_number)
            if not user_id:
                return {
                    "success": False,
                    "response": "User not found",
                    "session_data": session_data
                }
            session_data["user_id"] = user_id

        # Get or create session ID
        session_id = session_data.get("session_id")
        if not session_id:
            session_id = f"{user_id}_{int(time.time())}"
            session_data["session_id"] = session_id
            
            # Initialize session in database if needed
            try:
                session_info = {
                    "id": session_id,
                    "user_id": user_id,
                    "phone_number": phone_number,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                    "status": "active",
                    "messages": "[]",
                    "metadata": {}
                }
                supabase.table('sessions').insert(session_info).execute()
                logger.info(f"New session created in database: {session_id}")
            except Exception as e:
                logger.error(f"Error creating session in database: {str(e)}")

        # Get user context if not already in session
        if "user_context" not in session_data:
            user_context = get_user_context(user_id)
            session_data["user_context"] = user_context
        else:
            user_context = session_data["user_context"]

        # Check for call intention
        call_intention_score = detect_call_intention(message)
        logger.info(f"Call intention score: {call_intention_score}")
        
        # Check for intro request intention
        intro_intention_score = detect_intro_request_intention(message)
        logger.info(f"Intro request intention score: {intro_intention_score}")
        
        if call_intention_score > 0.8:
            # Handle call intention
            user_context_for_call = get_user_context_for_call(phone_number)
            user_name = user_context_for_call.get("name", "").split()[0] if user_context_for_call.get("name") else ""
            notification_message = generate_call_confirmation_message(user_name, user_context_for_call)
            
            # Store message
            store_message(user_id, phone_number, message, 'incoming')
            # Store message in outgoing and database
            store_message(user_id, phone_number, notification_message, 'outgoing')
            
            # Add messages to session history
            if "messages" not in session_data:
                session_data["messages"] = []
                
            session_data["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            session_data["messages"].append({
                "role": "assistant",
                "content": notification_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Update last activity
            session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
            
            # Update Redis if available
            if redis_client:
                session_key = f"session:{phone_number}"
                redis_client.setex(session_key, 60, json.dumps(session_data))
            
            # Update database
            try:
                supabase.table('sessions').update({
                    "last_activity": session_data["last_activity"],
                    "messages": json.dumps(session_data["messages"])
                }).eq("id", session_id).execute()
            except Exception as e:
                logger.error(f"Error updating session in database: {str(e)}")
            
            # Initiate call (this will happen asynchronously)
            try:
                call_result = make_vapi_outbound_call(phone_number, user_context_for_call)
                logger.info(f"VAPI call result: {call_result}")
            except Exception as e:
                logger.error(f"Error initiating VAPI call: {str(e)}")
            
            return {
                "success": True,
                "response": notification_message,
                "session_data": session_data,
                "call_initiated": True
            }
        
        elif intro_intention_score > 0.7:  # Nouveau: Gestion des demandes d'introduction
            # Handle intro request
            user_name = user_context.get('personal_profile', {}).get('name', "").split()[0] if user_context.get('personal_profile', {}).get('name') else ""
            confirmation_message = handle_intro_request(user_id, phone_number, user_name)
            
            # Store messages
            store_message(user_id, phone_number, message, 'incoming')
            store_message(user_id, phone_number, confirmation_message, 'outgoing')
            
            # Add messages to session history
            if "messages" not in session_data:
                session_data["messages"] = []
                
            session_data["messages"].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            session_data["messages"].append({
                "role": "assistant",
                "content": confirmation_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Update last activity
            session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
            
            # Update Redis if available
            if redis_client:
                session_key = f"session:{phone_number}"
                redis_client.setex(session_key, 60, json.dumps(session_data))
            
            # Update database
            try:
                supabase.table('sessions').update({
                    "last_activity": session_data["last_activity"],
                    "messages": json.dumps(session_data["messages"])
                }).eq("id", session_id).execute()
            except Exception as e:
                logger.error(f"Error updating session in database: {str(e)}")
            
            return {
                "success": True,
                "response": confirmation_message,
                "session_data": session_data,
                "intro_requested": True
            }
        
        # Process normal message
        response = process_message_with_context(message, user_context, phone_number)
        
        # Store messages
        store_message(user_id, phone_number, message, 'incoming')
        store_message(user_id, phone_number, response, 'outgoing')
        
        # Add messages to session history
        if "messages" not in session_data:
            session_data["messages"] = []
            
        session_data["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        session_data["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Update last activity
        session_data["last_activity"] = datetime.now(timezone.utc).isoformat()
        
        # Update Redis if available
        if redis_client:
            session_key = f"session:{phone_number}"
            redis_client.setex(session_key, 60, json.dumps(session_data))
        
        # Update database
        try:
            supabase.table('sessions').update({
                "last_activity": session_data["last_activity"],
                "messages": json.dumps(session_data["messages"])
            }).eq("id", session_id).execute()
        except Exception as e:
            logger.error(f"Error updating session in database: {str(e)}")
        
        return {
            "success": True,
            "response": response,
            "session_data": session_data
        }
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "response": "Sorry, an error occurred while processing your message.",
            "session_data": session_data
        }

def handle_intro_request(user_id: str, phone_number: str, user_name: str = "") -> str:
    """Traite une demande d'introduction en déclenchant le matching et l'introduction"""
    try:
        logging.info(f"[INTRO_REQUEST] Début du traitement pour user {user_id}")
        
        # Message de confirmation immédiat
        confirmation_msg = f"Parfait{' ' + user_name if user_name else ''} ! Laisse-moi te trouver quelqu'un d'intéressant... 🔍"
        
        # Déclencher le processus de matching et introduction en arrière-plan
        import threading
        thread = threading.Thread(
            target=trigger_matching_and_intro_for_user,
            args=(user_id, phone_number, user_name)
        )
        thread.start()
        
        return confirmation_msg
        
    except Exception as e:
        logging.error(f"[INTRO_REQUEST] Erreur lors du traitement: {str(e)}")
        return "Désolé, je n'ai pas pu traiter ta demande pour le moment. Peux-tu réessayer ?"

def trigger_matching_and_intro_for_user(user_id: str, phone_number: str, user_name: str = ""):
    """Déclenche le matching puis l'introduction pour un utilisateur spécifique"""
    try:
        logging.info(f"[INTRO_REQUEST] Déclenchement matching pour user {user_id}")
        
        # 1. Déclencher le calcul des matchs pour ce user
        matching_url = os.getenv("MATCHING_FUNCTION_URL", "https://func-matching-calculator.azurewebsites.net/api/trigger-matching")
        
        matching_response = requests.post(
            matching_url, 
            json={"user_id": user_id},
            timeout=45
        )
        
        if matching_response.status_code != 200:
            logging.error(f"[INTRO_REQUEST] Erreur matching: {matching_response.text}")
            error_msg = "Désolé, je n'ai pas pu calculer tes matchs. Réessaie plus tard !"
            send_whatsapp_message(phone_number, error_msg)
            return
        
        logging.info(f"[INTRO_REQUEST] Matchs calculés avec succès pour user {user_id}")
        
        # 2. Attendre que les matchs soient traités
        time.sleep(45)  # Augmenté pour laisser le temps au LLM de juger
        
        # 3. Déclencher l'introduction (utiliser l'endpoint classique pour l'instant)
        intro_url = os.getenv("INTRODUCTION_FUNCTION_URL", "https://func-message-generation-jackie.azurewebsites.net/api/generate-introduction")
        
        intro_response = requests.post(
            intro_url,
            json={"user_id": user_id},
            timeout=60
        )
        
        if intro_response.status_code == 200:
            logging.info(f"[INTRO_REQUEST] Introduction envoyée avec succès pour user {user_id}")
            # Le message d'introduction a été envoyé directement par la fonction
        else:
            logging.error(f"[INTRO_REQUEST] Erreur introduction: {intro_response.text}")
            fallback_msg = f"J'ai trouvé des personnes intéressantes pour toi{' ' + user_name if user_name else ''}, mais je n'arrive pas à t'envoyer l'introduction maintenant. Réessaie dans quelques minutes !"
            send_whatsapp_message(phone_number, fallback_msg)
        
    except Exception as e:
        logging.error(f"[INTRO_REQUEST] Erreur générale: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Message d'erreur à l'utilisateur
        error_msg = f"Désolé{' ' + user_name if user_name else ''}, j'ai rencontré un problème technique. Peux-tu réessayer dans quelques minutes ?"
        try:
            send_whatsapp_message(phone_number, error_msg)
        except:
            pass