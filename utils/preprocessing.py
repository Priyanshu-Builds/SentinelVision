import cv2
import numpy as np

def preprocess_frame(frame, target_size=(64, 64)):
    """
    Preprocess a frame:
      - Resize to target_size.
      - Convert from BGR to RGB.
      - Normalize pixel values to [0, 1].
    """
    resized_frame = cv2.resize(frame, target_size)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame.astype("float32") / 255.0
    return normalized_frame

if __name__ == "__main__":
    sample_img = cv2.imread("sample.jpg")  # Replace with a valid image path
    processed_img = preprocess_frame(sample_img)
    print("Processed image shape:", processed_img.shape)
