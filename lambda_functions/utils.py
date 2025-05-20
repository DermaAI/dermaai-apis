import os
import json
import boto3
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
import pymongo
from pymongo import MongoClient
import base64

s3_client = boto3.client('s3')

def get_mongodb_client():
    """Create and return a MongoDB client with HIPAA-compliant configuration."""
    client = MongoClient(
        os.environ['MONGODB_URI'],
        tls=True,
        tlsAllowInvalidCertificates=False,
        serverSelectionTimeoutMS=5000
    )
    return client

def upload_to_s3(image_data, filename):
    """Upload image to S3 with encryption."""
    bucket = os.environ['IMAGE_BUCKET']
    key = f"uploads/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
    
    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=image_data,
        ServerSideEncryption='AES256'
    )
    return f"s3://{bucket}/{key}"

def load_model(model_name):
    """Load TFLite model from S3."""
    bucket = os.environ['MODEL_BUCKET']
    local_path = f"/tmp/{model_name}"
    
    s3_client.download_file(bucket, model_name, local_path)
    interpreter = tf.lite.Interpreter(model_path=local_path)
    interpreter.allocate_tensors()
    return interpreter

def process_image(image_data):
    """Process image for model input."""
    image = Image.open(BytesIO(image_data))
    image = np.asarray(image.resize((224, 224)), dtype=np.float32)[..., :3]
    image = np.expand_dims(image, 0)
    image = image/255
    return np.vstack([image])

def create_response(prediction, confidence, image_path, model_type):
    """Create standardized API response."""
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "prediction": str(prediction),
            "confidence": str(confidence),
            "image_path": image_path,
            "model_type": model_type,
            "timestamp": datetime.utcnow().isoformat()
        })
    }

def store_prediction(prediction_data):
    """Store prediction results in MongoDB with HIPAA compliance."""
    client = get_mongodb_client()
    db = client.dermaai
    collection = db.predictions
    
    # Add HIPAA-compliant metadata
    prediction_data['created_at'] = datetime.utcnow()
    prediction_data['hipaa_compliant'] = True
    
    result = collection.insert_one(prediction_data)
    return str(result.inserted_id) 