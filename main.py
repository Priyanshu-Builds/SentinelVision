import cv2
import numpy as np
from model.optimized_cnn import optimized_model
from utils.preprocessing import preprocess_frame
from utils.adaptive_storage import save_key_frame
from utils.aws_integration import log_to_cloudwatch

def main():
    # Load the optimized model
    model = optimized_model(input_shape=(64, 64, 3))
    # Optionally load pre-trained weights:
    # model.load_weights('path_to_weights.h5')

    cap = cv2.VideoCapture(0)  # Default webcam
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    previous_saved_frame = None  # For adaptive storage

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame: resize to 64x64, convert BGR->RGB, normalize
        preprocessed = preprocess_frame(frame, target_size=(64, 64))
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension

        # Run inference using the optimized CNN model
        prediction = model.predict(preprocessed)
        threat_detected = prediction[0][0] > 0.5

        # Log threat to AWS CloudWatch if detected
        if threat_detected:
            log_to_cloudwatch("Threat detected in current frame.")

        # Adaptive storage: save frame only if it significantly differs from previous saved frame
        if threat_detected:
            previous_saved_frame, saved_filename = save_key_frame(frame, previous_saved_frame, threshold=5000)
            if saved_filename:
                print(f"Key frame saved: {saved_filename}")

        # Overlay the detection result on the frame
        label = "Threat" if threat_detected else "Normal"
        color = (0, 0, 255) if threat_detected else (0, 255, 0)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("SentinelVision - Live Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
