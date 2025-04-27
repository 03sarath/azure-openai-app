from flask import Flask
from flask_session import Session
from config import Config

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    Session(app)
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main.bp)
    
    return app 