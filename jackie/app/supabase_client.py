from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL_DEV")
SUPABASE_KEY = os.getenv("SUPABASE_KEY_DEV")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Variables d'environnement SUPABASE_URL et SUPABASE_KEY requises")

# Configuration sans paramètres proxy
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)