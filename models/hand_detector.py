import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import urllib.request
import os


class HandDetector:
    """Wrapper class for MediaPipe hand detection using Tasks API"""

    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Download model if not exists
        self.model_path = 'hand_landmarker.task'
        if not os.path.exists(self.model_path):
            self._download_model()

        # Initialize detector for VIDEO mode (synchronous processing)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0

    def _download_model(self):
        """Download the hand landmarker model"""
        print("Downloading hand landmarker model...")
        model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        try:
            urllib.request.urlretrieve(model_url, self.model_path)
            print("✓ Model downloaded successfully!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def find_hands(self, frame, draw=True):
        """
        Detect hands in frame and optionally draw landmarks

        Args:
            frame: BGR image from camera
            draw: Whether to draw landmarks on frame

        Returns:
            frame: Frame with drawings (if draw=True)
            results: Detection results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        # Detect hands
        results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp)
        self.frame_timestamp += 1

        if draw and results.hand_landmarks:
            frame = self._draw_landmarks(frame, results)

        return frame, results

    def _draw_landmarks(self, bgr_frame, detection_result):
        """Draw hand landmarks on the BGR frame"""
        annotated_frame = np.copy(bgr_frame)

        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks as circles
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_frame.shape[1])
                y = int(landmark.y * annotated_frame.shape[0])
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17)  # Palm
            ]

            for connection in connections:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                start_point = (int(start.x * annotated_frame.shape[1]),
                               int(start.y * annotated_frame.shape[0]))
                end_point = (int(end.x * annotated_frame.shape[1]),
                             int(end.y * annotated_frame.shape[0]))

                cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)

        return annotated_frame

    def get_landmarks(self, results):
        """
        Extract landmark coordinates as numpy array

        Returns:
            List of arrays, one per detected hand
            Each array has shape (21, 3) for x, y, z coordinates
        """
        landmarks_list = []

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                coords = []
                for landmark in hand_landmarks:
                    coords.append([landmark.x, landmark.y, landmark.z])
                landmarks_list.append(np.array(coords))

        return landmarks_list

    def get_handedness(self, results):
        """
        Get handedness (Left/Right) for each detected hand

        Returns:
            List of strings: ['Left', 'Right', ...]
        """
        handedness_list = []

        if results.handedness:
            for handedness in results.handedness:
                # handedness is a list of Classification objects
                label = handedness[0].category_name  # 'Left' or 'Right'
                handedness_list.append(label)

        return handedness_list

    def close(self):
        """Release resources"""
        self.landmarker.close()


if __name__ == "__main__":
    # Quick test
    print("Hand Detector Test")
    print("Press 'q' to quit")

    detector = HandDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame, draw=True)
        landmarks = detector.get_landmarks(results)
        handedness = detector.get_handedness(results)

        if landmarks:
            info = f"Hands: {len(landmarks)}"
            if handedness:
                info += f" ({', '.join(handedness)})"
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Detector Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✓ Test completed")