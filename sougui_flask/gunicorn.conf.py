# gunicorn.conf.py
# Lancement : gunicorn -c gunicorn.conf.py "app:create_app()"

bind        = "0.0.0.0:5000"
workers     = 2          # (2 × CPU) + 1 recommandé
threads     = 2
timeout     = 120        # Prophet peut être lent au 1er appel
worker_class = "sync"
preload_app  = True      # Charge les modèles une seule fois
loglevel    = "info"
accesslog   = "-"        # stdout
errorlog    = "-"        # stdout
