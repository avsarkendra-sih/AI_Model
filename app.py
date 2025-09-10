import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.api.routes import api_bp
from src.utils.logger import setup_logger
from config.config import Config

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Setup logging
    logger = setup_logger()
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "message": "AvsarKendra AI API is running",
            "version": "0.1.0"
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error"}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
