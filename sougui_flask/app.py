"""
╔══════════════════════════════════════════════════════════════╗
║  Sougui — Dashboard Intelligent ML                          ║
║  Application Flask principale                               ║
╚══════════════════════════════════════════════════════════════╝
"""

from flask import Flask
from config import get_config
from routes.dashboard  import dashboard_bp
from routes.prediction import prediction_bp
from routes.api        import api_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(get_config())

    # ── Blueprints ────────────────────────────────────────────────
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp, url_prefix='/predict')
    app.register_blueprint(api_bp,        url_prefix='/api')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
