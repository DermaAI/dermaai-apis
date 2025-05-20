# DermaAI API - Serverless Architecture

A HIPAA-compliant serverless API for skin disease detection using AWS Lambda, S3, and MongoDB.

## Architecture

- **AWS Lambda**: Serverless functions for ML model inference
- **AWS S3**: Secure storage for images and ML models
- **MongoDB Atlas**: HIPAA-compliant database for storing predictions
- **API Gateway**: REST API endpoints
- **AWS Systems Manager Parameter Store**: Secure storage for sensitive configuration

## Security Features

- HIPAA compliance through MongoDB Atlas
- Server-side encryption for S3 objects
- TLS encryption for all data in transit
- Secure parameter storage
- CORS configuration
- Input validation
- Error handling

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
aws configure
```

3. Set up MongoDB Atlas:
- Create a HIPAA-compliant cluster
- Configure network access
- Create a database user
- Store the connection string in AWS Parameter Store

4. Deploy the serverless application:
```bash
serverless deploy
```

5. Upload ML models to S3:
```bash
aws s3 cp models/main-model1.tflite s3://dermaai-models/
aws s3 cp models/second-model.tflite s3://dermaai-models/
```

## API Endpoints

### Main Prediction
- **POST** `/predict/main`
- **Body**: 
  ```json
  {
    "image": "base64_encoded_image",
    "filename": "optional_filename.jpg"
  }
  ```

### Cancer Prediction
- **POST** `/predict/cancer`
- **Body**: 
  ```json
  {
    "image": "base64_encoded_image",
    "filename": "optional_filename.jpg"
  }
  ```

## Response Format
```json
{
  "prediction": "prediction_class",
  "confidence": "confidence_score",
  "image_path": "s3_path",
  "model_type": "main|cancer",
  "timestamp": "ISO_timestamp"
}
```

## Development

1. Install the Serverless Framework:
```bash
npm install -g serverless
```

2. Install Python requirements:
```bash
pip install -r requirements.txt
```

3. Local testing:
```bash
serverless invoke local -f predictMain --path test/event.json
```

## Security Considerations

- All images are encrypted at rest in S3
- MongoDB connection uses TLS
- API Gateway has CORS configured
- Lambda functions have minimal IAM permissions
- All sensitive data is stored in Parameter Store
- Regular security audits and monitoring
- HIPAA compliance maintained throughout the pipeline 