"""
Real-time sign language recognition with text building
Works with BOTH CNN and Random Forest models automatically
"""

import cv2
import pickle
import numpy as np
import os
from collections import deque
from datetime import datetime

# Import hand detector
import sys
sys.path.insert(0, 'E:/translation_system/models')
from hand_detector import HandDetector


class SignLanguageRecognizer:
    """Real-time sign language recognition with text building"""
    
    def __init__(self):
        """Load model and initialize"""
        print("Loading Sign Language Recognizer...")
        
        # Load hand detector
        self.detector = HandDetector()
        
        # Try to load CNN model first, then Random Forest
        self.model_type = None
        
        # Try CNN
        if os.path.exists('models/sign_classifier_cnn.keras'):
            print("→ Loading CNN model...")
            import tensorflow as tf
            self.model = tf.keras.models.load_model('models/sign_classifier_cnn.keras')
            
            with open('models/sign_classifier_cnn_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            
            self.model_type = 'cnn'
            
        # Try Random Forest
        elif os.path.exists('models/sign_classifier_rf.pkl'):
            print("→ Loading Random Forest model...")
            with open('models/sign_classifier_rf.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                metadata = data
            
            self.model_type = 'random_forest'
        
        else:
            print("⚠ No model found!")
            print("Please run train_model_smart.py first.")
            exit()
        
        # Load metadata
        self.label_encoder = metadata['label_encoder']
        self.max_len = metadata['max_len']
        self.classes = metadata['classes']
        
        print(f"✓ {self.model_type.upper()} model loaded (Accuracy: {metadata['accuracy']*100:.1f}%)")
        print(f"  Classes: {self.classes}")
        
        # Buffers
        self.frame_buffer = deque(maxlen=self.max_len)
        self.prediction_buffer = deque(maxlen=5)
        
        # Text building
        self.current_text = ""
        self.current_word = ""
        
        # Sign hold detection
        self.last_sign = None
        self.same_sign_count = 0
        self.hold_threshold = 25  # ~1 second at 30 FPS
        
        # Settings
        self.confidence_threshold = 0.7
        
        print("✓ Recognizer ready!")
    
    
    def predict(self, sequence):
        """Make prediction with current model"""
        if self.model_type == 'cnn':
            # CNN expects 3D input
            sequence_reshaped = np.expand_dims(sequence, axis=0)
            predictions = self.model.predict(sequence_reshaped, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
        else:  # random_forest
            # Random Forest expects flattened input
            sequence_flat = sequence.flatten().reshape(1, -1)
            pred_class = self.model.predict(sequence_flat)[0]
            
            # Get confidence from probability
            probabilities = self.model.predict_proba(sequence_flat)[0]
            confidence = np.max(probabilities)
            
            # Convert label to encoded format
            pred_class = self.label_encoder.transform([pred_class])[0]
        
        # Decode prediction
        predicted_sign = self.label_encoder.inverse_transform([pred_class])[0]
        
        return predicted_sign, confidence
    
    
    def add_letter(self, letter):
        """Add letter to current word"""
        if letter != 'nothing':
            self.current_word += letter
    
    
    def finish_word(self):
        """Finish current word"""
        if self.current_word:
            if self.current_text:
                self.current_text += " "
            self.current_text += self.current_word
            self.current_word = ""
    
    
    def save_text(self):
        """Save text to file"""
        if self.current_text or self.current_word:
            full_text = self.current_text
            if self.current_word:
                full_text += " " + self.current_word
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"output_text_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(full_text)
            
            print(f"\n✓ Text saved to {filename}")
    
    
    def run(self):
        """Main recognition loop"""
        print("\n" + "=" * 70)
        print("SIGN LANGUAGE RECOGNIZER - TEXT BUILDER")
        print(f"Model: {self.model_type.upper()}")
        print("=" * 70)
        print("\nControls:")
        print("  - Hold sign for 1 second to add letter")
        print("  - Press SPACE to finish word")
        print("  - Press BACKSPACE to delete letter")
        print("  - Press 'c' to clear all text")
        print("  - Press 'q' to quit (auto-saves)")
        print()
        
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("⚠ Cannot open camera!")
            return
        
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Detect hands
            frame, results = self.detector.find_hands(frame)
            landmarks = self.detector.get_landmarks(results)

            # Process if hand detected
            if landmarks and len(landmarks) > 0:
                # Take first hand only
                hand = landmarks[0]

                # Flatten/normalize like in training
                hand_flat = np.array(hand)  # shape (21, 3)
                wrist = hand_flat[0]
                hand_flat = hand_flat - wrist
                max_val = np.max(np.abs(hand_flat))
                if max_val > 0:
                    hand_flat = hand_flat / max_val
                hand_flat = hand_flat.flatten()  # shape (63,)

                self.frame_buffer.append(hand_flat)
                
                # Make prediction when buffer full
                if len(self.frame_buffer) == self.max_len:
                    # Prepare sequence
                    sequence = np.array(list(self.frame_buffer))
                    
                    # Predict
                    predicted_sign, confidence = self.predict(sequence)
                    
                    # Add to buffer
                    self.prediction_buffer.append(predicted_sign)
                    
                    # Smooth prediction (majority vote)
                    if len(self.prediction_buffer) >= 5:
                        # Count votes
                        votes = {}
                        for pred in self.prediction_buffer:
                            votes[pred] = votes.get(pred, 0) + 1
                        
                        final_prediction = max(votes, key=votes.get)
                        
                        # Display
                        if confidence >= self.confidence_threshold:
                            # Hold detection
                            if final_prediction == self.last_sign:
                                self.same_sign_count += 1
                                
                                # Progress bar for hold
                                progress = int((self.same_sign_count / self.hold_threshold) * 100)
                                bar_width = int((progress / 100) * 200)
                                cv2.rectangle(frame, (10, 100), (210, 130), (50, 50, 50), -1)
                                cv2.rectangle(frame, (10, 100), (10 + bar_width, 130), (0, 255, 0), -1)
                                cv2.putText(frame, f"{progress}%", (220, 125),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                if self.same_sign_count >= self.hold_threshold:
                                    self.add_letter(final_prediction)
                                    self.same_sign_count = 0
                                    
                                    # Flash feedback
                                    cv2.putText(frame, f"ADDED: {final_prediction}", 
                                              (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                                              1.2, (0, 255, 0), 3)
                            else:
                                self.last_sign = final_prediction
                                self.same_sign_count = 0
                            
                            # Show prediction
                            cv2.putText(frame, f"Sign: {final_prediction} ({confidence*100:.0f}%)",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"Uncertain ({confidence*100:.0f}%)",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                # No hand
                cv2.putText(frame, "No hand detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                self.frame_buffer.clear()
                self.prediction_buffer.clear()
                self.last_sign = None
                self.same_sign_count = 0
            
            # Display current word
            if self.current_word:
                cv2.putText(frame, f"Current: {self.current_word}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display full text
            if self.current_text:
                text_display = self.current_text
                if len(text_display) > 50:
                    text_display = "..." + text_display[-47:]
                
                cv2.rectangle(frame, (5, h-50), (w-5, h-10), (0, 0, 0), -1)
                cv2.putText(frame, text_display, (10, h-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Sign Language Recognizer', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                self.save_text()
                break
            elif key == ord(' '):
                self.finish_word()
            elif key == 8:  # Backspace
                if self.current_word:
                    self.current_word = self.current_word[:-1]
            elif key == ord('c'):
                self.current_word = ""
                self.current_text = ""
        
        camera.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run()
