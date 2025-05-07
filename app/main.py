from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse
import os
import logging
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Configuration Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Configuration Twilio
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialisation du client OpenAI
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def send_whatsapp_message(to_number: str, message: str) -> dict:
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
            logger.info(f"Message WhatsApp envoyé avec succès à {to_number}")
            return response.json()
        else:
            logger.error(f"Erreur lors de l'envoi du message WhatsApp: {response.status_code}")
            logger.error(f"Réponse: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi du message WhatsApp: {str(e)}")
        return None

def generate_response(message: str, phone_number: str) -> str:
    """Génère une réponse via Azure OpenAI"""
    try:
        logger.info(f"Génération de réponse pour le message: {message[:50]}...")
        
        system_prompt = """
        You are Jackie, an AI social connector who communicates via WhatsApp. You're friendly, 
        warm, and engaging. Your job is to connect with people, understand them, and help them 
        meet interesting people. Respond in a conversational, casual style using short messages 
        appropriate for WhatsApp. Use emojis sparingly. Keep your responses concise and natural.
        """
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de réponse: {str(e)}")
        return "Je suis désolé, je ne peux pas traiter votre message pour le moment. Veuillez réessayer plus tard."

@app.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    From: str = Form(...),
    Body: str = Form(...),
    SmsMessageSid: str = Form(None)
):
    try:
        logger.info(f"Message WhatsApp reçu de {From}: {Body[:50]}...")
        
        # Extraire le numéro de téléphone du format Twilio
        phone_number = From.replace('whatsapp:', '')
        
        # Générer une réponse
        response_text = generate_response(Body, phone_number)
        
        # Envoyer la réponse via l'API Twilio
        send_result = send_whatsapp_message(phone_number, response_text)
        
        # Répondre à Twilio avec un TwiML vide pour éviter d'envoyer un message de réponse automatique
        # Puisque nous répondons via l'API Twilio directement
        return PlainTextResponse(content="", status_code=200)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook: {str(e)}")
        return PlainTextResponse(content="", status_code=500)

@app.get("/health")
def health_check():
    """Endpoint pour vérifier que l'application est en ligne"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}