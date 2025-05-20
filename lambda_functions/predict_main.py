import json
import base64
from utils import (
    load_model,
    process_image,
    upload_to_s3,
    create_response,
    store_prediction
)

def handler(event, context):
    try:
        # Parse the request body
        body = json.loads(event['body'])
        image_data = base64.b64decode(body['image'])
        filename = body.get('filename', 'image.jpg')
        
        # Upload image to S3
        image_path = upload_to_s3(image_data, filename)
        
        # Load and run model
        interpreter = load_model('main-model1.tflite')
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Process image and get prediction
        processed_image = process_image(image_data)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        prediction = int(np.argmax(output_data))
        confidence = float(np.amax(output_data))
        
        # Store prediction in MongoDB
        prediction_data = {
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence,
            'model_type': 'main',
            'filename': filename
        }
        prediction_id = store_prediction(prediction_data)
        
        # Create response
        return create_response(
            prediction=prediction,
            confidence=confidence,
            image_path=image_path,
            model_type='main'
        )
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        } 