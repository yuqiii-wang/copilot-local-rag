import os
from urllib.parse import quote_plus

# Manual .env loader since python-dotenv might not be installed
def load_env(env_path):
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key not in os.environ:
                        os.environ[key] = value

# Load .env from the same directory as this file
basedir = os.path.abspath(os.path.dirname(__file__))
load_env(os.path.join(basedir, '.env'))

class Config:
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "repo_ask")
    # Note: These are the app credentials, not necessarily the admin credentials needed for init
    POSTGRES_USER = os.getenv("POSTGRES_USER", "action_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "P@33w0rd123")
    POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "repo_ask")
    
    # Pool settings
    POSTGRES_MIN_POOL_SIZE = int(os.getenv("POSTGRES_MIN_POOL_SIZE", 1))
    POSTGRES_MAX_POOL_SIZE = int(os.getenv("POSTGRES_MAX_POOL_SIZE", 10))

    @property
    def DATABASE_URL(self):
        encoded_user = quote_plus(self.POSTGRES_USER)
        encoded_password = quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{encoded_user}:{encoded_password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"

config = Config()
