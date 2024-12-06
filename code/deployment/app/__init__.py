from flask import Flask
import os


def create_app():
    app = Flask(__name__, template_folder=os.path.join(
        os.path.dirname(__file__), '..', 'templates'))

    with app.app_context():
        # Import parts of the application
        from . import routes

        # Register Blueprints or Routes
        app.register_blueprint(routes.main)

    return app
