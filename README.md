
# Sentinel Vision: AI-Driven Night Vision Surveillance with Intelligent Motion Analytics and Adaptive Storage

SentinelVision is a smart surveillance system that leverages an optimized, lightweight CNN for real-time threat detection on resource-constrained devices. The project includes modules for video capture, image preprocessing, threat detection, adaptive storage of key frames, and AWS integration (for S3 uploads and CloudWatch logging). An AWS Lambda function is also provided for dynamic resolution scaling of incoming images.

## Suspicious activity detection

<img width="1112" height="626" alt="image" src="https://github.com/user-attachments/assets/512691c7-2060-408a-b116-6bdfc53d9518" />

## Adaptive Storage

<img width="704" height="590" alt="image" src="https://github.com/user-attachments/assets/f6e84aeb-3271-4447-89c0-5698567ddb6e" />

## Implementation

<img width="773" height="634" alt="image" src="https://github.com/user-attachments/assets/182fb3b4-6cda-41b4-b429-cb4cdeaf0b73" />

## Project Structure

```
SentinelVision/
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── main.py               # Main script: video capture, frame processing, threat detection, adaptive storage, AWS logging
├── lambda_function.py    # AWS Lambda function for dynamic resolution scaling
├── model/
│   └── optimized_cnn.py  # Optimized CNN model definition
├── utils/
│   ├── preprocessing.py  # Preprocessing functions for images
│   ├── aws_integration.py  # AWS integration functions (S3 upload, CloudWatch logging)
│   └── adaptive_storage.py  # Functions for adaptive storage (saving key frames)
└── experiments/
    └── run_experiment.py # Script for training and evaluating the model
```

## Features

- **Optimized Lightweight CNN Model:**  
  Uses depthwise separable convolutions, ReLU6 activation, Batch Normalization, Global Average Pooling, Dropout, and L2 regularization for efficient threat detection.

- **Real-Time Threat Detection:**  
  Captures video frames, preprocesses them, and runs inference in real time to classify frames as "Threat" or "Normal."

- **Adaptive Storage:**  
  Saves key frames only when they differ significantly from previously saved frames, reducing storage costs.

- **AWS Integration:**  
  Includes functions for uploading files to AWS S3 and logging events to AWS CloudWatch Logs.

- **AWS Lambda for Dynamic Resolution Scaling:**  
  A sample Lambda function that adjusts image resolution based on a motion level parameter, useful for adaptive video processing in the cloud.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Priyanshu-Builds/SentinelVision.git
   cd SentinelVision
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Real-Time Detection

To start the real-time threat detection, run:
```bash
python main.py
```
This script will:
- Capture video from your default webcam.
- Preprocess each frame.
- Run the optimized CNN model to classify frames as "Threat" or "Normal."
- Log threat events to AWS CloudWatch (if detected).
- Use adaptive storage to save only key frames.
- Display the live video feed with overlays.

### Training and Evaluation

To train and evaluate the model using dummy data (or modify the script to use your dataset), run:
```bash
python experiments/run_experiment.py
```

### AWS Lambda Function

The `lambda_function.py` file is an AWS Lambda function that performs dynamic resolution scaling on an input image based on a provided motion level. To deploy:
- Package `lambda_function.py` with its dependencies.
- Deploy it using the AWS Lambda console, CLI, or through infrastructure as code.
- Ensure your event payload includes `image_data` (base64-encoded image) and `motion_level`.

### AWS Integration Utilities

The AWS integration functions are in `utils/aws_integration.py`. Use:
- `upload_to_s3(file_path, bucket_name, object_name)` to upload files.
- `log_to_cloudwatch(message)` to log messages.

### Adaptive Storage Utilities

The adaptive storage function in `utils/adaptive_storage.py` saves a frame only if it is significantly different from the previous saved frame. This helps to reduce redundant storage of similar frames.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements

- Thanks to the research community for inspiring lightweight CNN architectures.
