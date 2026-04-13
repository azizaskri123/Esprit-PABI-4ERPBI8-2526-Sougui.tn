"""
config.py
──────────
Configuration Flask selon l'environnement.
Usage dans app.py :  app.config.from_object(config)
"""

import os


class Config:
    """Configuration de base."""
    SECRET_KEY  = os.environ.get('SECRET_KEY', 'sougui-ml-dev-secret-change-in-prod')
    MODELS_DIR  = os.environ.get('MODELS_DIR', os.path.join(os.path.dirname(__file__), 'models'))
    DEBUG       = False
    TESTING     = False


class DevelopmentConfig(Config):
    """Développement local."""
    DEBUG = True


class ProductionConfig(Config):
    """Production (Gunicorn)."""
    DEBUG = False
    # SECRET_KEY doit être défini via variable d'environnement


config_map = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'default':     DevelopmentConfig,
}

def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config_map.get(env, DevelopmentConfig)
