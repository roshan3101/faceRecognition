from flask import Flask, request, jsonify
from recognizer import FaceRecognizer
import os
import logging
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv

app = Flask(__name__)
recognizer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

mongo_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("MONGODB_DB_NAME")

def initialize_recognizer(mongo_uri: str, db_name: str):
    """Initialize the face recognizer with MongoDB connection."""
    global recognizer
    try:
        recognizer = FaceRecognizer(mongo_uri, db_name)
        recognizer.initialize_models()
        recognizer.connect_to_mongodb()
        recognizer.precompute_embeddings()
        return True
    except Exception as e:
        logger.error(f"Error initializing recognizer: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if recognizer is None:
            return jsonify({
                "status": "error",
                "message": "Recognizer not initialized"
            }), 503
        
        # Check MongoDB connection
        recognizer.collection.find_one({})
        
        # Check if models are loaded
        if recognizer.face_detector is None or recognizer.face_recognizer is None:
            return jsonify({
                "status": "error",
                "message": "Models not loaded"
            }), 503
        
        return jsonify({
            "status": "healthy",
            "message": "All systems operational",
            "users_loaded": len(recognizer.embeddings_cache)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503

@app.route('/train', methods=['POST'])
def train():
    """Train endpoint to update embeddings."""
    try:
        if recognizer is None:
            return jsonify({
                "status": "error",
                "message": "Recognizer not initialized"
            }), 503
        
        recognizer.precompute_embeddings()
        
        return jsonify({
            "status": "success",
            "message": "Training completed",
            "users_loaded": len(recognizer.embeddings_cache)
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize face from uploaded image."""
    try:
        if recognizer is None:
            return jsonify({
                "status": "error",
                "message": "Recognizer not initialized"
            }), 503
        
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No selected file"
            }), 400
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process the image
            user_id = recognizer.recognize_face(temp_path)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            if user_id is None:
                return jsonify({
                    "status": "success",
                    "message": "No match found",
                    "userId": None
                })
            
            return jsonify({
                "status": "success",
                "message": "Match found",
                "userId": user_id
            })
            
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    # Get MongoDB URI from environment variable
    mongo_uri = os.getenv('MONGODB_URI')
    if not mongo_uri:
        logger.error("MONGODB_URI environment variable not set")
        exit(1)
    
    # Initialize the recognizer
    if not initialize_recognizer(mongo_uri, db_name):
        logger.error("Failed to initialize recognizer")
        exit(1)
    
    # Start the Flask server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 