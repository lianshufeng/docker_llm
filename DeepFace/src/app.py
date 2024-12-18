# 3rd parth dependencies
# project dependencies
from deepface import DeepFace
# from deepface.api.src.modules.core.routes import blueprint
from deepface.commons.logger import Logger
from flask import Flask
from flask_cors import CORS

from modules.core.routes import blueprint

logger = Logger()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(blueprint)
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app
