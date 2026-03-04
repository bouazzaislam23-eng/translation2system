"""
Real-time sign language recognition with NLP and Text-to-Speech
Enhanced version with word suggestions, autocorrect, and voice output
"""

import cv2
import pickle
import numpy as np
import os
from collections import deque
from datetime import datetime
import sys

# NLP and TTS imports
try:
    import nltk
    from nltk.corpus import words
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠ NLTK not available - word suggestions disabled")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠ pyttsx3 not available - text-to-speech disabled")

# Import hand detector
sys.path.insert(0, 'E:/translation_system/models')
from hand_detector import HandDetector


class NLPProcessor:
    """Natural Language Processing for word suggestions and autocorrect"""
    
    def __init__(self):
        """Initialize NLP components"""
        self.word_list = set()
        self.enabled = NLTK_AVAILABLE
        
        if NLTK_AVAILABLE:
            try:
                # Download words corpus if not present
                try:
                    nltk.data.find('corpora/words')
                except LookupError:
                    print("Downloading NLTK words corpus...")
                    nltk.download('words', quiet=True)
                
                # Load word list
                self.word_list = set(word.lower() for word in words.words())
                print(f"✓ NLP initialized with {len(self.word_list):,} words")
            except Exception as e:
                print(f"⚠ NLP initialization failed: {e}")
                self.enabled = False
        else:
            print("⚠ NLP features disabled (NLTK not installed)")
    
    
    def get_suggestions(self, partial_word, max_suggestions=3):
        """Get word suggestions based on partial input"""
        if not self.enabled or not partial_word:
            return []
        
        partial_lower = partial_word.lower()
        
        # Find words starting with partial input
        suggestions = [
            word for word in self.word_list 
            if word.startswith(partial_lower)
        ]
        
        # Sort by length (shorter words are more common)
        suggestions.sort(key=len)
        
        return suggestions[:max_suggestions]
    
    
    def autocorrect(self, word):
        """Simple autocorrect - check if word exists, suggest correction"""
        if not self.enabled or not word:
            return word
        
        word_lower = word.lower()
        
        # Word is correct
        if word_lower in self.word_list:
            return word
        
        # Find closest match using edit distance
        closest = self._find_closest(word_lower)
        
        return closest if closest else word
    
    
    def _find_closest(self, word, max_distance=2):
        """Find closest word using simple edit distance"""
        # Simple approach: check words with similar length
        target_len = len(word)
        candidates = [
            w for w in self.word_list 
            if abs(len(w) - target_len) <= max_distance
        ]
        
        # Find best match
        best_match = None
        best_distance = float('inf')
        
        for candidate in candidates[:1000]:  # Limit search for speed
            distance = self._edit_distance(word, candidate)
            if distance < best_distance:
                best_distance = distance
                best_match = candidate
        
        return best_match if best_distance <= max_distance else None
    
    
    def _edit_distance(self, s1, s2):
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    
    def is_valid_word(self, word):
        """Check if word exists in dictionary"""
        if not self.enabled or not word:
            return False
        
        return word.lower() in self.word_list


class TextToSpeech:
    """Text-to-speech engine for reading recognized text"""
    
    def __init__(self):
        """Initialize TTS engine"""
        self.enabled = TTS_AVAILABLE

        if TTS_AVAILABLE:
            try:
                # Test initialization
                test_engine = pyttsx3.init()
                test_engine.stop()
                del test_engine

                print("✓ Text-to-Speech initialized")
            except Exception as e:
                print(f"⚠ TTS initialization failed: {e}")
                self.enabled = False
        else:
            print("⚠ TTS disabled (pyttsx3 not installed)")


    def speak(self, text):
        """Speak the given text - creates fresh engine each time"""
        if not self.enabled or not text:
            return

        try:
            # Create fresh engine for each speech (ensures repeatability)
            engine = pyttsx3.init()

            # Configure voice properties
            engine.setProperty('rate', 150)  # Speed
            engine.setProperty('volume', 0.9)  # Volume

            # Speak text
            engine.say(text)
            engine.runAndWait()

            # Clean up
            engine.stop()
            del engine

        except Exception as e:
            print(f"⚠ TTS error: {e}")


class SignLanguageRecognizer:
    """Real-time sign language recognition with NLP and TTS"""

    def __init__(self):
        """Load model and initialize"""
        print("\n" + "=" * 70)
        print("SIGN LANGUAGE RECOGNIZER - NLP + TTS EDITION")
        print("=" * 70)

        # Load hand detector
        self.detector = HandDetector()

        # Load model
        self.model_type = None

        if os.path.exists('models/sign_classifier_cnn.keras'):
            print("→ Loading CNN model...")
            import tensorflow as tf
            self.model = tf.keras.models.load_model('models/sign_classifier_cnn.keras')

            with open('models/sign_classifier_cnn_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)

            self.model_type = 'cnn'

        elif os.path.exists('models/sign_classifier_rf.pkl'):
            print("→ Loading Random Forest model...")
            with open('models/sign_classifier_rf.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                metadata = data

            self.model_type = 'random_forest'

        else:
            print("⚠ No model found!")
            print("Please run train_full_alphabet.py first.")
            exit()

        # Load metadata
        self.label_encoder = metadata['label_encoder']
        self.max_len = metadata['max_len']
        self.classes = metadata['classes']

        print(f"✓ {self.model_type.upper()} model loaded")
        print(f"  Accuracy: {metadata['accuracy']*100:.1f}%")
        print(f"  Classes: {len(self.classes)}")

        # Initialize NLP and TTS
        print("\n→ Initializing NLP and TTS...")
        self.nlp = NLPProcessor()
        self.tts = TextToSpeech()

        # Buffers
        self.frame_buffer = deque(maxlen=self.max_len)
        self.prediction_buffer = deque(maxlen=5)

        # Text building
        self.current_text = ""
        self.current_word = ""
        self.word_suggestions = []

        # Sign hold detection
        self.last_sign = None
        self.same_sign_count = 0
        self.hold_threshold = 25

        # Auto word-break
        self.frames_without_hand = 0
        self.word_break_threshold = 60  # 2 seconds

        # Settings
        self.confidence_threshold = 0.7
        self.show_suggestions = True
        self.auto_correct = True

        print("\n✓ System ready!")


    def predict(self, sequence):
        """Make prediction with current model"""
        if self.model_type == 'cnn':
            sequence_reshaped = np.expand_dims(sequence, axis=0)
            predictions = self.model.predict(sequence_reshaped, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

        else:
            sequence_flat = sequence.flatten().reshape(1, -1)
            pred_class = self.model.predict(sequence_flat)[0]
            probabilities = self.model.predict_proba(sequence_flat)[0]
            confidence = np.max(probabilities)
            pred_class = self.label_encoder.transform([pred_class])[0]

        predicted_sign = self.label_encoder.inverse_transform([pred_class])[0]

        return predicted_sign, confidence


    def add_letter(self, letter):
        """Add letter to current word"""
        if letter == 'space':
            self.finish_word()
        elif letter == 'del':
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.update_suggestions()
        elif letter != 'nothing':
            self.current_word += letter.upper()
            self.update_suggestions()


    def update_suggestions(self):
        """Update word suggestions based on current word"""
        if self.show_suggestions and self.nlp.enabled:
            self.word_suggestions = self.nlp.get_suggestions(self.current_word, max_suggestions=3)
        else:
            self.word_suggestions = []


    def finish_word(self):
        """Finish current word with optional autocorrect"""
        if self.current_word:
            word_to_add = self.current_word

            # Apply autocorrect
            if self.auto_correct and self.nlp.enabled:
                corrected = self.nlp.autocorrect(word_to_add)
                if corrected != word_to_add.lower():
                    print(f"  Autocorrected: {word_to_add} → {corrected.upper()}")
                    word_to_add = corrected.upper()

            # Add to text
            if self.current_text:
                self.current_text += " "
            self.current_text += word_to_add

            # Reset
            self.current_word = ""
            self.word_suggestions = []


    def use_suggestion(self, index):
        """Use a word suggestion"""
        if 0 <= index < len(self.word_suggestions):
            self.current_word = self.word_suggestions[index].upper()
            self.finish_word()


    def speak_text(self):
        """Speak current text"""
        if self.tts.enabled:
            full_text = self.current_text
            if self.current_word:
                full_text += " " + self.current_word

            if full_text.strip():
                print("🔊 Speaking...")
                self.tts.speak(full_text)
            else:
                print("⚠ No text to speak")
        else:
            print("⚠ Text-to-speech not available")


    def save_text(self):
        """Save text to file"""
        if self.current_text or self.current_word:
            full_text = self.current_text
            if self.current_word:
                full_text += " " + self.current_word

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"output_text_{timestamp}.txt"

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(full_text)

            print(f"\n✓ Text saved to {filename}")


    def run(self):
        """Main recognition loop"""
        print("\n" + "=" * 70)
        print("CONTROLS")
        print("=" * 70)
        print("Sign Recognition:")
        print("  - Hold sign for 1 second to add letter")
        print("  - No hand for 2 seconds = auto finish word")
        print("\nKeyboard Controls:")
        print("  SPACE     - Finish word manually")
        print("  BACKSPACE - Delete last letter")
        print("  1/2/3     - Use word suggestion 1/2/3")
        print("  s         - Speak text (Text-to-Speech)")
        print("  a         - Toggle autocorrect")
        print("  w         - Toggle word suggestions")
        print("  c         - Clear all text")
        print("  q         - Quit and save")
        print("=" * 70 + "\n")

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
                self.frames_without_hand = 0

                # Normalize landmarks
                hand = landmarks[0]
                hand_flat = np.array(hand)
                wrist = hand_flat[0]
                hand_flat = hand_flat - wrist
                max_val = np.max(np.abs(hand_flat))
                if max_val > 0:
                    hand_flat = hand_flat / max_val
                hand_flat = hand_flat.flatten()

                self.frame_buffer.append(hand_flat)

                # Make prediction when buffer full
                if len(self.frame_buffer) == self.max_len:
                    sequence = np.array(list(self.frame_buffer))
                    predicted_sign, confidence = self.predict(sequence)

                    self.prediction_buffer.append(predicted_sign)

                    # Smooth prediction
                    if len(self.prediction_buffer) >= 5:
                        votes = {}
                        for pred in self.prediction_buffer:
                            votes[pred] = votes.get(pred, 0) + 1

                        final_prediction = max(votes, key=votes.get)

                        if confidence >= self.confidence_threshold:
                            # Hold detection
                            if final_prediction == self.last_sign:
                                self.same_sign_count += 1

                                # Progress bar
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
                # No hand detected
                self.frames_without_hand += 1

                # Auto finish word after 2 seconds without hand
                if self.frames_without_hand >= self.word_break_threshold:
                    if self.current_word:
                        self.finish_word()
                        cv2.putText(frame, "AUTO WORD BREAK", (10, 180),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    self.frames_without_hand = 0

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

            # Display word suggestions
            if self.word_suggestions and self.show_suggestions:
                y_pos = 140
                cv2.putText(frame, "Suggestions:", (w - 250, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                for i, suggestion in enumerate(self.word_suggestions[:3], 1):
                    y_pos += 25
                    cv2.putText(frame, f"{i}: {suggestion.upper()}", (w - 250, y_pos),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Display settings indicators
            settings_y = h - 80
            if self.auto_correct:
                cv2.putText(frame, "AC:ON", (w - 100, settings_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if self.show_suggestions:
                cv2.putText(frame, "SUG:ON", (w - 100, settings_y + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display full text
            if self.current_text:
                text_display = self.current_text
                if len(text_display) > 50:
                    text_display = "..." + text_display[-47:]

                cv2.rectangle(frame, (5, h-50), (w-5, h-10), (0, 0, 0), -1)
                cv2.putText(frame, text_display, (10, h-20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Sign Language Recognizer - NLP + TTS', frame)

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
                    self.update_suggestions()
            elif key == ord('c'):
                self.current_word = ""
                self.current_text = ""
                self.word_suggestions = []
            elif key == ord('s'):
                self.speak_text()
            elif key == ord('a'):
                self.auto_correct = not self.auto_correct
                status = "ON" if self.auto_correct else "OFF"
                print(f"Autocorrect: {status}")
            elif key == ord('w'):
                self.show_suggestions = not self.show_suggestions
                status = "ON" if self.show_suggestions else "OFF"
                print(f"Word suggestions: {status}")
                if not self.show_suggestions:
                    self.word_suggestions = []
            elif key == ord('1'):
                self.use_suggestion(0)
            elif key == ord('2'):
                self.use_suggestion(1)
            elif key == ord('3'):
                self.use_suggestion(2)

        camera.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run()