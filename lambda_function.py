import json
import base64
import cv2
import numpy as np

def lambda_handler(event, context):
    """
    AWS Lambda function for dynamic resolution scaling.
    It decodes a base64-encoded image, resizes it based on a motion level parameter,
    and returns the processed image as a base64 string.
    
    Expected event keys:
      - "image_data": Base64 encoded JPEG image.
      - "motion_level": A float (0 to 1) indicating the detected motion intensity.
    """
    image_data = event.get("image_data")
    if not image_data:
        return {
            "statusCode": 400,
            "body": json.dumps("No image_data provided")
        }
    
    # Decode base64 to image
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Determine target resolution based on motion_level
    motion_level = event.get("motion_level", 0)
    if motion_level > 0.5:
        # High motion: retain original resolution
        target_size = (img.shape[1], img.shape[0])
    else:
        # Low motion: downscale to half resolution
        target_size = (img.shape[1] // 2, img.shape[0] // 2)
    
    # Resize image
    scaled_img = cv2.resize(img, target_size)
    
    # Encode the image to JPEG and then to base64
    ret, buffer = cv2.imencode('.jpg', scaled_img)
    if not ret:
        return {
            "statusCode": 500,
            "body": json.dumps("Failed to encode image")
        }
    processed_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "statusCode": 200,
        "body": json.dumps({"processed_image": processed_base64})
    }
