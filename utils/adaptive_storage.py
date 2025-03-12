import cv2
import numpy as np
import os

def save_key_frame(frame, previous_frame, threshold=5000, save_dir="key_frames"):
    """
    Save the current frame if it differs significantly from the previous saved frame.
    
    :param frame: Current frame (numpy array)
    :param previous_frame: Previous saved frame (numpy array) or None
    :param threshold: Difference threshold (sum of absolute differences)
    :param save_dir: Directory to save key frames
    :return: Updated previous frame and filename if saved (or None)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # If there's no previous frame, save this one.
    if previous_frame is None:
        filename = f"{save_dir}/frame_{int(cv2.getTickCount())}.jpg"
        cv2.imwrite(filename, frame)
        return frame, filename

    # Compute the sum of absolute differences between the frames.
    diff = cv2.absdiff(frame, previous_frame)
    diff_sum = np.sum(diff)
    
    if diff_sum > threshold:
        filename = f"{save_dir}/frame_{int(cv2.getTickCount())}.jpg"
        cv2.imwrite(filename, frame)
        return frame, filename
    else:
        return previous_frame, None

if __name__ == "__main__":
    # Test with sample images (ensure sample1.jpg and sample2.jpg exist)
    img1 = cv2.imread("sample1.jpg")
    img2 = cv2.imread("sample2.jpg")
    prev_frame, filename = save_key_frame(img2, img1, threshold=5000)
    if filename:
        print("Saved key frame as:", filename)
    else:
        print("Frame difference below threshold; not saved.")
