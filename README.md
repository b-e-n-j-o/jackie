# Jackie API

API de chat pour Jackie, le connecteur social IA basé sur FastAPI.

## Structure du projet

```
jackie-bot/
├── app/
│   ├── main.py              ← point d'entrée FastAPI
│   ├── chat_logic.py        ← logique de chat (sessions, LangChain, etc.)
│   ├── models.py            ← modèles Pydantic
│   ├── supabase_client.py   ← client Supabase
├── requirements.txt
├── .env
└── README.md
```
Oui

## Installation

1. Cloner le repository
2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Créer un fichier `.env` avec les variables d'environnement suivantes :
```
SUPABASE_URL=votre_url_supabase
SUPABASE_KEY=votre_clé_supabase
AZURE_OPENAI_ENDPOINT=votre_endpoint_azure
AZURE_OPENAI_API_KEY=votre_clé_api_azure
AZURE_OPENAI_DEPLOYMENT_NAME=votre_déploiement_azure
```

## Démarrage

Pour démarrer le serveur de développement :

```bash
uvicorn app.main:app --reload
```

Le serveur sera accessible à l'adresse : http://localhost:8000

## API Endpoints

### GET /
Vérifie si l'API est en cours d'exécution.

### POST /chat
Endpoint principal pour le chat.

Corps de la requête :
```json
{
    "phone_number": "+33612345678",
    "message": "Bonjour Jackie!"
}
```

### POST /chat/raw
Version alternative de l'endpoint de chat qui accepte un corps JSON brut.

## Documentation API

La documentation interactive de l'API est disponible aux adresses suivantes :
- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc 