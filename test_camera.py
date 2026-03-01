"""
Test script to verify camera and hand tracking setup
Press 'q' to quit
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import urllib.request


def draw_landmarks_on_image(bgr_frame, detection_result):
    """Draw hand landmarks on the image"""
    annotated_frame = np.copy(bgr_frame)

    for hand_landmarks in detection_result.hand_landmarks:
        # Draw landmarks
        for landmark in hand_landmarks:
            x = int(landmark.x * annotated_frame.shape[1])
            y = int(landmark.y * annotated_frame.shape[0])
            cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

    return annotated_frame


def main():
    # Setup Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    latest_result = {'landmarks': None}

    def result_callback(result, output_image, timestamp_ms):
        latest_result['landmarks'] = result

    # Download model
    model_path = 'hand_landmarker.task'
    model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

    print("Verifying model file...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("✓ Model ready!")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Configure hand landmarker
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=result_callback
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Camera started! Press 'q' to quit")

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_timestamp = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

            # Process frame
            landmarker.detect_async(mp_image, frame_timestamp)
            frame_timestamp += 1

            # Draw results
            if latest_result['landmarks'] and latest_result['landmarks'].hand_landmarks:
                frame = draw_landmarks_on_image(frame, latest_result['landmarks'])
                num_hands = len(latest_result['landmarks'].hand_landmarks)
                cv2.putText(frame, f"Hands detected: {num_hands}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Hand Tracking - Press Q to Quit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Camera closed")


if __name__ == "__main__":
    main()