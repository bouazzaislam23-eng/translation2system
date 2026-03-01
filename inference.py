"""
Real-time sign language translation
Uses trained model to recognize signs from webcam
"""

import cv2
import numpy as np
import pickle
import os
from collections import deque
from models.hand_detector import HandDetector


class SignLanguageTranslator:
    """Real-time sign language recognition"""

    def __init__(self, model_path='models/sign_classifier.pkl'):
        # Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first!")

        print("Loading trained model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.max_len = model_data['max_len']
        self.classes = model_data['classes']

        print(f"✓ Model loaded!")
        print(f"  Recognized signs: {self.classes}")

        # Initialize hand detector
        self.detector = HandDetector()

        # Buffer to collect sequences
        self.sequence_length = 30  # Same as training
        self.sequence_buffer = deque(maxlen=self.sequence_length)

        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)  # Last 5 predictions

        # Display settings
        self.current_prediction = "None"
        self.confidence = 0.0

    def predict_sign(self):
        """Predict sign from current sequence buffer"""
        if len(self.sequence_buffer) < self.sequence_length:
            return None, 0.0

        # Convert buffer to feature vector
        features = []
        for frame_data in self.sequence_buffer:
            if frame_data is not None:
                features.extend(frame_data.flatten())

        # Pad to match training data length
        if len(features) < self.max_len:
            features.extend([0] * (self.max_len - len(features)))
        else:
            features = features[:self.max_len]

        # Predict
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]

        # Get confidence (probability)
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = np.max(probabilities)

        return prediction, confidence

    def get_smoothed_prediction(self, prediction, confidence):
        """Smooth predictions to avoid flickering"""
        if confidence > 0.6:  # Only add high-confidence predictions
            self.prediction_buffer.append(prediction)

        if len(self.prediction_buffer) >= 3:
            # Most common prediction in buffer
            most_common = max(set(self.prediction_buffer),
                              key=list(self.prediction_buffer).count)
            return most_common

        return prediction if confidence > 0.6 else "..."

    def run(self):
        """Start real-time translation"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot access camera")
            return

        print("\n" + "=" * 60)
        print("REAL-TIME SIGN LANGUAGE TRANSLATOR")
        print("=" * 60)
        print("Instructions:")
        print("  - Perform signs in front of the camera")
        print("  - Hold each sign for 1-2 seconds")
        print("  - Press 'q' to quit")
        print("  - Press 'c' to clear prediction buffer")
        print("=" * 60 + "\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, results = self.detector.find_hands(frame, draw=True)

            # Get landmarks
            landmarks = self.detector.get_landmarks(results)

            # Add to sequence buffer
            if landmarks:
                # Use first hand
                self.sequence_buffer.append(landmarks[0])
            else:
                # No hand detected
                self.sequence_buffer.append(None)

            # Make prediction if buffer is full
            if len(self.sequence_buffer) == self.sequence_length:
                prediction, confidence = self.predict_sign()

                if prediction:
                    smoothed = self.get_smoothed_prediction(prediction, confidence)
                    self.current_prediction = smoothed
                    self.confidence = confidence

            # Draw UI
            self._draw_ui(frame, landmarks)

            cv2.imshow('Sign Language Translator - Press Q to Quit', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear buffers
                self.sequence_buffer.clear()
                self.prediction_buffer.clear()
                self.current_prediction = "None"
                print("✓ Buffers cleared")

        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("\n✓ Translator closed")

    def _draw_ui(self, frame, landmarks):
        """Draw user interface on frame"""
        height, width = frame.shape[:2]

        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "Sign Language Translator", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Prediction
        color = (0, 255, 0) if self.confidence > 0.6 else (0, 165, 255)
        cv2.putText(frame, f"Sign: {self.current_prediction}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Confidence
        cv2.putText(frame, f"Confidence: {self.confidence * 100:.1f}%", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Buffer status
        buffer_fill = len(self.sequence_buffer)
        buffer_text = f"Buffer: {buffer_fill}/{self.sequence_length}"
        cv2.putText(frame, buffer_text, (width - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Hand detection status
        status = "Hand Detected" if landmarks else "No Hand"
        status_color = (0, 255, 0) if landmarks else (0, 0, 255)
        cv2.putText(frame, status, (width - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # Known signs
        y_offset = height - 100
        cv2.putText(frame, "Known signs:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        signs_text = ", ".join(self.classes)
        cv2.putText(frame, signs_text, (20, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == "__main__":
    try:
        translator = SignLanguageTranslator()
        translator.run()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run these steps first:")
        print("1. python utils/data_collector.py  (collect training data)")
        print("2. python train_model.py          (train the model)")
        print("3. python inference.py            (run real-time translation)")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()