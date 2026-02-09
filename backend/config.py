import os

def load_env(env_path):
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    if key not in os.environ:
                        os.environ[key] = value

basedir = os.path.abspath(os.path.dirname(__file__))
load_env(os.path.join(basedir, '.env'))

class Config:
    SCORE_NORMALIZATION_FACTOR = 100.0
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'

config = Config()
