# Face Recognition API

A production-ready face recognition system using YOLO and InsightFace.

## Features

- Face detection using YOLO
- Face recognition using InsightFace
- MongoDB integration for user data
- REST API endpoints for training and recognition
- Health check endpoint

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file from `.env.example` and set your MongoDB URI

5. Run the development server:
   ```bash
   python face_recognition_api.py
   ```

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following environment variables:
   - `MONGODB_URI`: Your MongoDB connection string
   - `PORT`: 5000

4. Deploy!

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /train`: Train the model with new data
- `POST /recognize`: Recognize a face from an uploaded image

## Environment Variables

- `MONGODB_URI`: MongoDB connection string
- `PORT`: Port to run the server on (default: 5000)

## Production Considerations

- The API uses Gunicorn with gevent workers for production
- Memory usage is optimized for Render's standard plan
- Automatic scaling is configured for optimal performance 