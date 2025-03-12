import boto3
import logging
import time

# Initialize AWS clients (ensure AWS credentials are configured)
s3_client = boto3.client('s3')
logs_client = boto3.client('logs')

# CloudWatch configuration
LOG_GROUP = "SentinelVisionLogs"
LOG_STREAM = "MotionDetectionStream"

def upload_to_s3(file_path, bucket_name, object_name):
    """
    Upload a file to an AWS S3 bucket.
    """
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Uploaded {file_path} to S3 bucket {bucket_name} as {object_name}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

def log_to_cloudwatch(message):
    """
    Log a message to AWS CloudWatch Logs.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(message)
    try:
        timestamp = int(round(time.time() * 1000))
        logs_client.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=LOG_STREAM,
            logEvents=[{
                'timestamp': timestamp,
                'message': message
            }],
        )
    except Exception as e:
        print(f"Error logging to CloudWatch: {e}")

if __name__ == "__main__":
    # Test AWS functions (ensure valid bucket name and test file)
    upload_to_s3("test.txt", "your-s3-bucket-name", "test.txt")
    log_to_cloudwatch("Test log message from SentinelVision AWS integration")
