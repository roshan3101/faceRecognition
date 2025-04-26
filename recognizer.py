import os
import sys
import json
import requests
import numpy as np
import cv2
import torch
import insightface
from pymongo import MongoClient
from scipy.spatial.distance import cosine
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dotenv import load_dotenv
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class FaceRecognizer:
    def __init__(self, mongo_uri: str, db_name: str="test", collection_name: str = "users", yolo_model_path: str = "yolov11l-face.pt"):
        """Initialize the face recognizer with MongoDB connection and models."""
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.yolo_model_path = yolo_model_path
        self.face_detector = None
        self.face_recognizer = None
        self.embeddings_cache: Dict[str, List[np.ndarray]] = {}
        self.similarity_threshold = 0.5
        
    def initialize_models(self):
        """Initialize YOLO and InsightFace models."""
        try:
            # Initialize YOLO for face detection
            self.face_detector = YOLO(self.yolo_model_path)
            logger.info("YOLO model initialized successfully")
            
            # Initialize InsightFace for face recognition
            self.face_recognizer = insightface.model_zoo.get_model('buffalo_l')
            self.face_recognizer.prepare(ctx_id=0)
            logger.info("InsightFace model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def connect_to_mongodb(self):
        """Establish connection to MongoDB Atlas."""
        try:
            logger.info(f"Attempting to connect to MongoDB at: {self.mongo_uri}")
            client = MongoClient(self.mongo_uri)
            
            # Test the connection
            client.admin.command('ping')
            logger.info("Successfully pinged MongoDB server")
            
            # Get database and collection
            db = client[self.db_name]
            self.collection = db[self.collection_name]
            
            # Test collection access
            count = self.collection.count_documents({})
            logger.info(f"Successfully connected to collection. Found {count} documents")
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def download_image(self, url: str) -> np.ndarray:
        """Download image from URL and convert to numpy array."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect and crop faces from an image using YOLO."""
        try:
            # Run YOLO inference
            results = self.face_detector(image)
            
            cropped_faces = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Add padding around the face (20% on each side)
                    h, w = image.shape[:2]
                    face_width = x2 - x1
                    face_height = y2 - y1
                    padding_x = int(face_width * 0.2)
                    padding_y = int(face_height * 0.2)
                    
                    # Apply padding with boundary checks
                    x1 = max(0, x1 - padding_x)
                    y1 = max(0, y1 - padding_y)
                    x2 = min(w, x2 + padding_x)
                    y2 = min(h, y2 + padding_y)
                    
                    # Crop face with padding
                    cropped_face = image[y1:y2, x1:x2]
                    
                    if cropped_face.size > 0:  # Ensure we have a valid crop
                        # Make sure the face is large enough
                        if cropped_face.shape[0] >= 32 and cropped_face.shape[1] >= 32:
                            cropped_faces.append(cropped_face)
            
            return cropped_faces
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            raise

    def get_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Generate face embedding using InsightFace."""
        try:
            # Preprocess the face image
            # Resize to required size (112x112 for InsightFace)
            face_image = cv2.resize(face_image, (112, 112))
            
            # Convert to RGB if needed (InsightFace expects RGB)
            if len(face_image.shape) == 2:  # If grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 3:  # If BGR
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get embedding using InsightFace
            embedding = self.face_recognizer.get_feat(face_image)
            return embedding
        except Exception as e:
            logger.error(f"Error generating face embedding: {str(e)}")
            raise

    def precompute_embeddings(self):
        """Precompute and cache face embeddings for all users."""
        try:
            # Log MongoDB connection details
            logger.info(f"Connecting to MongoDB: {self.mongo_uri}")
            logger.info(f"Database: {self.db_name}, Collection: {self.collection_name}")
            
            # Get all users
            users = list(self.collection.find({}))
            logger.info(f"Found {len(users)} users in the database")
            
            if not users:
                logger.warning("No users found in the database")
                return
            
            for user in users:
                user_id = str(user['_id'])
                logger.info(f"Processing user: {user_id}")
                
                # Check if user has faceData
                if 'faceData' not in user:
                    logger.warning(f"User {user_id} has no faceData field")
                    continue
                
                face_images = user['faceData'].get('faceImages', [])
                logger.info(f"User {user_id} has {len(face_images)} face images")
                
                if not face_images:
                    logger.warning(f"User {user_id} has no face images")
                    continue
                
                embeddings = []
                for image_data in face_images:
                    try:
                        # Extract URL from the image data object
                        image_url = image_data.get('url')
                        if not image_url:
                            logger.warning(f"No URL found in image data: {image_data}")
                            continue
                            
                        logger.info(f"Processing image: {image_url}")
                        image = self.download_image(image_url)
                        
                        # Log image details
                        logger.info(f"Downloaded image shape: {image.shape}")
                        
                        faces = self.detect_faces(image)
                        logger.info(f"Detected {len(faces)} faces in image")
                        
                        for face in faces:
                            embedding = self.get_face_embedding(face)
                            embeddings.append(embedding)
                            logger.info(f"Generated embedding of shape: {embedding.shape}")
                            
                    except Exception as e:
                        logger.error(f"Error processing image for user {user_id}: {str(e)}")
                        continue
                
                if embeddings:
                    self.embeddings_cache[user_id] = embeddings
                    logger.info(f"Cached {len(embeddings)} embeddings for user {user_id}")
                else:
                    logger.warning(f"No embeddings generated for user {user_id}")
            
            logger.info(f"Precomputed embeddings for {len(self.embeddings_cache)} users")
            if self.embeddings_cache:
                total_embeddings = sum(len(embeddings) for embeddings in self.embeddings_cache.values())
                logger.info(f"Total embeddings cached: {total_embeddings}")
            else:
                logger.warning("No embeddings were cached")
                
        except Exception as e:
            logger.error(f"Error precomputing embeddings: {str(e)}")
            raise

    def find_best_match(self, input_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the best matching user ID based on cosine similarity."""
        best_match_id = None
        best_similarity = -1
        
        for user_id, embeddings in self.embeddings_cache.items():
            for stored_embedding in embeddings:
                similarity = 1 - cosine(input_embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = user_id
        
        if best_similarity < self.similarity_threshold:
            return None, best_similarity
        
        return best_match_id, best_similarity

    def recognize_face(self, image_path: str) -> Optional[str]:
        """Main function to recognize a face from an image path."""
        try:
            # Load and process input image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Detect faces
            faces = self.detect_faces(image)
            if not faces:
                logger.warning("No faces detected in the input image")
                return None
            
            # Get embedding for the first detected face
            input_embedding = self.get_face_embedding(faces[0])
            
            # Find best match
            user_id, similarity = self.find_best_match(input_embedding)
            
            if user_id is None:
                logger.info(f"No match found (best similarity: {similarity:.4f})")
                return None
            
            logger.info(f"Found match: user {user_id} with similarity {similarity:.4f}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python recognizer.py <mongo_uri> <image_path>")
        sys.exit(1)
    
    mongo_uri = sys.argv[1]
    image_path = sys.argv[2]
    
    try:
        recognizer = FaceRecognizer(mongo_uri)
        recognizer.initialize_models()
        recognizer.connect_to_mongodb()
        recognizer.precompute_embeddings()
        
        user_id = recognizer.recognize_face(image_path)
        print(json.dumps({"userId": user_id}))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main() 