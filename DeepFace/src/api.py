import argparse

from deepface import modules, DeepFace
from deepface.api.src.modules.core.routes import blueprint
from deepface.commons.logger import Logger
from flask import Flask
from flask_cors import CORS

from modules.core.routes import blueprint2

logger = Logger()


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(blueprint)
    app.register_blueprint(blueprint2)



    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    return app


if __name__ == "__main__":
    deepface_app = create_app()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port of serving api")
    args = parser.parse_args()
    deepface_app.run(host="0.0.0.0", port=args.port)
