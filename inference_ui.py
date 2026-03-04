"""
Sign Language Recognition - Beautiful GUI Interface
Single window with camera feed + all information panels
"""

import cv2
import pickle
import numpy as np
import os
from collections import deque
from datetime import datetime
import sys
import threading
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, font

# NLP and TTS imports
try:
    import nltk
    from nltk.corpus import words
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Import hand detector
sys.path.insert(0, 'E:/translation_system/models')
from hand_detector import HandDetector


class NLPProcessor:
    """Natural Language Processing"""

    def __init__(self):
        self.word_list = set()
        self.enabled = NLTK_AVAILABLE

        if NLTK_AVAILABLE:
            try:
                try:
                    nltk.data.find('corpora/words')
                except LookupError:
                    nltk.download('words', quiet=True)

                self.word_list = set(word.lower() for word in words.words())
            except:
                self.enabled = False

    def get_suggestions(self, partial_word, max_suggestions=3):
        if not self.enabled or not partial_word:
            return []

        partial_lower = partial_word.lower()
        suggestions = [w for w in self.word_list if w.startswith(partial_lower)]
        suggestions.sort(key=len)
        return suggestions[:max_suggestions]

    def autocorrect(self, word):
        if not self.enabled or not word:
            return word

        word_lower = word.lower()
        if word_lower in self.word_list:
            return word

        # Simple autocorrect
        target_len = len(word)
        candidates = [w for w in self.word_list if abs(len(w) - target_len) <= 2]

        best_match = None
        best_distance = float('inf')

        for candidate in candidates[:500]:
            distance = sum(c1 != c2 for c1, c2 in zip(word_lower, candidate))
            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        return best_match if best_distance <= 2 else word


class TextToSpeech:
    """Text-to-speech"""

    def __init__(self):
        self.enabled = TTS_AVAILABLE

    def speak(self, text):
        if not self.enabled or not text:
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except:
            pass


class BeautifulGUI:
    """Beautiful GUI window with camera and info panels"""

    def __init__(self):
        """Initialize GUI"""
        self.root = tk.Tk()
        self.root.title("🤟 Sign Language Recognition System")
        self.root.configure(bg='#1e1e2e')

        # Colors
        self.colors = {
            'bg': '#1e1e2e',
            'panel': '#2b2b3c',
            'accent1': '#89b4fa',
            'accent2': '#f38ba8',
            'accent3': '#a6e3a1',
            'text': '#cdd6f4',
            'text_dim': '#6c7086'
        }

        # Fonts
        self.font_title = font.Font(family="Segoe UI", size=16, weight="bold")
        self.font_large = font.Font(family="Segoe UI", size=24, weight="bold")
        self.font_normal = font.Font(family="Segoe UI", size=12)
        self.font_small = font.Font(family="Segoe UI", size=10)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""

        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Camera feed
        left_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Camera label
        self.camera_label = tk.Label(left_frame, bg='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Right side - Info panels
        right_frame = tk.Frame(main_frame, bg=self.colors['bg'], width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            right_frame,
            text="Sign Language\nRecognition",
            font=self.font_title,
            bg=self.colors['bg'],
            fg=self.colors['accent1'],
            justify=tk.CENTER
        )
        title_label.pack(pady=(0, 20))

        # Recognition panel
        self.create_recognition_panel(right_frame)

        # Current word panel
        self.create_current_word_panel(right_frame)

        # Suggestions panel
        self.create_suggestions_panel(right_frame)

        # Complete text panel
        self.create_text_panel(right_frame)

        # Statistics panel
        self.create_stats_panel(right_frame)

        # Controls panel
        self.create_controls_panel(right_frame)

    def create_panel(self, parent, title, height=None):
        """Create a styled panel"""
        frame = tk.Frame(parent, bg=self.colors['panel'], highlightbackground=self.colors['accent1'], highlightthickness=2)
        frame.pack(fill=tk.X, pady=(0, 10))
        if height:
            frame.configure(height=height)

        # Title
        title_label = tk.Label(
            frame,
            text=title,
            font=self.font_normal,
            bg=self.colors['panel'],
            fg=self.colors['accent1']
        )
        title_label.pack(anchor=tk.W, padx=10, pady=(5, 0))

        # Content frame
        content = tk.Frame(frame, bg=self.colors['panel'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        return content

    def create_recognition_panel(self, parent):
        """Create recognition info panel"""
        content = self.create_panel(parent, "📡 Recognition")

        # Sign
        sign_frame = tk.Frame(content, bg=self.colors['panel'])
        sign_frame.pack(fill=tk.X, pady=2)

        tk.Label(sign_frame, text="Sign:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).pack(side=tk.LEFT)
        self.sign_label = tk.Label(sign_frame, text="None", font=self.font_normal, bg=self.colors['panel'], fg=self.colors['text'])
        self.sign_label.pack(side=tk.LEFT, padx=(5, 0))

        # Confidence
        conf_frame = tk.Frame(content, bg=self.colors['panel'])
        conf_frame.pack(fill=tk.X, pady=2)

        tk.Label(conf_frame, text="Confidence:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).pack(side=tk.LEFT)
        self.conf_label = tk.Label(conf_frame, text="0%", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text'])
        self.conf_label.pack(side=tk.LEFT, padx=(5, 0))

        # Progress bar
        self.conf_progress = ttk.Progressbar(content, length=300, mode='determinate')
        self.conf_progress.pack(fill=tk.X, pady=2)

        # Hold progress
        hold_frame = tk.Frame(content, bg=self.colors['panel'])
        hold_frame.pack(fill=tk.X, pady=2)

        tk.Label(hold_frame, text="Hold:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).pack(side=tk.LEFT)
        self.hold_label = tk.Label(hold_frame, text="0%", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text'])
        self.hold_label.pack(side=tk.LEFT, padx=(5, 0))

        self.hold_progress = ttk.Progressbar(content, length=300, mode='determinate')
        self.hold_progress.pack(fill=tk.X, pady=2)

    def create_current_word_panel(self, parent):
        """Create current word panel"""
        content = self.create_panel(parent, "✏️ Current Word", height=80)

        self.current_word_label = tk.Label(
            content,
            text="",
            font=self.font_large,
            bg=self.colors['panel'],
            fg=self.colors['accent2']
        )
        self.current_word_label.pack(expand=True)

    def create_suggestions_panel(self, parent):
        """Create suggestions panel"""
        content = self.create_panel(parent, "💡 Suggestions")

        self.sug1_label = tk.Label(content, text="1. ", font=self.font_normal, bg=self.colors['panel'], fg=self.colors['text_dim'])
        self.sug1_label.pack(anchor=tk.W)

        self.sug2_label = tk.Label(content, text="2. ", font=self.font_normal, bg=self.colors['panel'], fg=self.colors['text_dim'])
        self.sug2_label.pack(anchor=tk.W)

        self.sug3_label = tk.Label(content, text="3. ", font=self.font_normal, bg=self.colors['panel'], fg=self.colors['text_dim'])
        self.sug3_label.pack(anchor=tk.W)

    def create_text_panel(self, parent):
        """Create complete text panel"""
        content = self.create_panel(parent, "📝 Complete Text", height=100)

        self.text_display = tk.Text(
            content,
            font=self.font_normal,
            bg=self.colors['panel'],
            fg=self.colors['accent3'],
            wrap=tk.WORD,
            height=3,
            relief=tk.FLAT
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)

    def create_stats_panel(self, parent):
        """Create statistics panel"""
        content = self.create_panel(parent, "📊 Statistics")

        stats_frame = tk.Frame(content, bg=self.colors['panel'])
        stats_frame.pack()

        # Letters
        tk.Label(stats_frame, text="Letters:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.letters_label = tk.Label(stats_frame, text="0", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text'])
        self.letters_label.grid(row=0, column=1, sticky=tk.W)

        # Words
        tk.Label(stats_frame, text="Words:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.words_label = tk.Label(stats_frame, text="0", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text'])
        self.words_label.grid(row=1, column=1, sticky=tk.W)

        # Corrections
        tk.Label(stats_frame, text="Corrections:", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text_dim']).grid(row=2, column=0, sticky=tk.W, padx=5)
        self.corrections_label = tk.Label(stats_frame, text="0", font=self.font_small, bg=self.colors['panel'], fg=self.colors['text'])
        self.corrections_label.grid(row=2, column=1, sticky=tk.W)

    def create_controls_panel(self, parent):
        """Create controls panel"""
        content = self.create_panel(parent, "⌨️ Controls")

        controls_text = """SPACE - Finish word
1/2/3 - Use suggestion
S - Speak text
A - Toggle autocorrect
C - Clear all
Q - Quit"""

        tk.Label(
            content,
            text=controls_text,
            font=self.font_small,
            bg=self.colors['panel'],
            fg=self.colors['text_dim'],
            justify=tk.LEFT
        ).pack(anchor=tk.W)

    def update_camera(self, frame):
        """Update camera feed"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit camera label
        h, w = rgb_frame.shape[:2]
        target_w = 640
        target_h = int(h * (target_w / w))
        rgb_frame = cv2.resize(rgb_frame, (target_w, target_h))

        # Convert to ImageTk
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

    def update_info(self, sign, confidence, hold_progress, current_word,
                   suggestions, text, stats):
        """Update all information panels"""

        # Recognition
        self.sign_label.configure(text=sign if sign else "None")
        self.conf_label.configure(text=f"{int(confidence*100)}%")
        self.conf_progress['value'] = confidence * 100
        self.hold_label.configure(text=f"{int(hold_progress*100)}%")
        self.hold_progress['value'] = hold_progress * 100

        # Current word
        self.current_word_label.configure(text=current_word if current_word else "")

        # Suggestions
        self.sug1_label.configure(text=f"1. {suggestions[0].upper()}" if len(suggestions) > 0 else "1. ")
        self.sug2_label.configure(text=f"2. {suggestions[1].upper()}" if len(suggestions) > 1 else "2. ")
        self.sug3_label.configure(text=f"3. {suggestions[2].upper()}" if len(suggestions) > 2 else "3. ")

        # Complete text
        self.text_display.delete('1.0', tk.END)
        self.text_display.insert('1.0', text)

        # Stats
        self.letters_label.configure(text=str(stats.get('letters', 0)))
        self.words_label.configure(text=str(stats.get('words', 0)))
        self.corrections_label.configure(text=str(stats.get('corrections', 0)))


class SignLanguageRecognizer:
    """Sign language recognizer with GUI"""

    def __init__(self):
        """Initialize"""
        print("Loading Sign Language Recognizer...")

        # Load model
        if os.path.exists('models/sign_classifier_cnn.keras'):
            import tensorflow as tf
            self.model = tf.keras.models.load_model('models/sign_classifier_cnn.keras')
            with open('models/sign_classifier_cnn_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            self.model_type = 'cnn'
        elif os.path.exists('models/sign_classifier_rf.pkl'):
            with open('models/sign_classifier_rf.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                metadata = data
            self.model_type = 'random_forest'
        else:
            print("⚠ No model found!")
            exit()

        self.label_encoder = metadata['label_encoder']
        self.max_len = metadata['max_len']

        print(f"✓ Model loaded ({metadata['accuracy']*100:.1f}% accuracy)")

        # Components
        self.detector = HandDetector()
        self.nlp = NLPProcessor()
        self.tts = TextToSpeech()
        self.gui = BeautifulGUI()

        # Buffers
        self.frame_buffer = deque(maxlen=self.max_len)
        self.prediction_buffer = deque(maxlen=5)

        # State
        self.current_sign = ""
        self.confidence = 0
        self.current_word = ""
        self.current_text = ""
        self.suggestions = []
        self.hold_progress = 0

        self.last_sign = None
        self.same_sign_count = 0
        self.hold_threshold = 25
        self.frames_without_hand = 0
        self.word_break_threshold = 60
        self.confidence_threshold = 0.7

        self.autocorrect_on = True
        self.suggestions_on = True

        self.stats = {'letters': 0, 'words': 0, 'corrections': 0}

        # Camera
        self.camera = cv2.VideoCapture(0)

        # Bind keyboard
        self.gui.root.bind('<space>', lambda e: self.finish_word())
        self.gui.root.bind('<BackSpace>', lambda e: self.backspace())
        self.gui.root.bind('s', lambda e: self.speak_text())
        self.gui.root.bind('c', lambda e: self.clear_all())
        self.gui.root.bind('a', lambda e: self.toggle_autocorrect())
        self.gui.root.bind('1', lambda e: self.use_suggestion(0))
        self.gui.root.bind('2', lambda e: self.use_suggestion(1))
        self.gui.root.bind('3', lambda e: self.use_suggestion(2))
        self.gui.root.bind('q', lambda e: self.quit())

        print("✓ System ready!\n")

    def predict(self, sequence):
        """Make prediction"""
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
        """Add letter"""
        if letter == 'space':
            self.finish_word()
        elif letter == 'del':
            if self.current_word:
                self.current_word = self.current_word[:-1]
                self.update_suggestions()
        elif letter != 'nothing':
            self.current_word += letter.upper()
            self.stats['letters'] += 1
            self.update_suggestions()

    def update_suggestions(self):
        """Update suggestions"""
        if self.suggestions_on and self.nlp.enabled:
            self.suggestions = self.nlp.get_suggestions(self.current_word, 3)
        else:
            self.suggestions = []

    def finish_word(self):
        """Finish word"""
        if self.current_word:
            word_to_add = self.current_word

            if self.autocorrect_on and self.nlp.enabled:
                corrected = self.nlp.autocorrect(word_to_add)
                if corrected != word_to_add.lower():
                    self.stats['corrections'] += 1
                    word_to_add = corrected.upper()

            if self.current_text:
                self.current_text += " "
            self.current_text += word_to_add

            self.current_word = ""
            self.suggestions = []
            self.stats['words'] += 1

    def backspace(self):
        """Backspace"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.update_suggestions()

    def clear_all(self):
        """Clear all text"""
        self.current_word = ""
        self.current_text = ""
        self.suggestions = []

    def toggle_autocorrect(self):
        """Toggle autocorrect"""
        self.autocorrect_on = not self.autocorrect_on
        print(f"Autocorrect: {'ON' if self.autocorrect_on else 'OFF'}")

    def use_suggestion(self, index):
        """Use suggestion"""
        if 0 <= index < len(self.suggestions):
            self.current_word = self.suggestions[index].upper()
            self.finish_word()

    def speak_text(self):
        """Speak text"""
        full_text = self.current_text
        if self.current_word:
            full_text += " " + self.current_word

        if full_text.strip():
            def speak_async():
                self.tts.speak(full_text)
            threading.Thread(target=speak_async, daemon=True).start()

    def quit(self):
        """Quit application"""
        self.camera.release()
        self.gui.root.quit()

    def update_frame(self):
        """Update one frame"""
        ret, frame = self.camera.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # Detect hands
        frame, results = self.detector.find_hands(frame)
        landmarks = self.detector.get_landmarks(results)

        if landmarks and len(landmarks) > 0:
            self.frames_without_hand = 0

            # Normalize
            hand = landmarks[0]
            hand_flat = np.array(hand)
            wrist = hand_flat[0]
            hand_flat = hand_flat - wrist
            max_val = np.abs(hand_flat).max()
            if max_val > 0:
                hand_flat = hand_flat / max_val
            hand_flat = hand_flat.flatten()

            self.frame_buffer.append(hand_flat)

            if len(self.frame_buffer) == self.max_len:
                sequence = np.array(list(self.frame_buffer))
                predicted_sign, confidence = self.predict(sequence)

                self.current_sign = predicted_sign
                self.confidence = confidence

                self.prediction_buffer.append(predicted_sign)

                if len(self.prediction_buffer) >= 5:
                    votes = {}
                    for pred in self.prediction_buffer:
                        votes[pred] = votes.get(pred, 0) + 1

                    final_prediction = max(votes, key=votes.get)

                    if confidence >= self.confidence_threshold:
                        if final_prediction == self.last_sign:
                            self.same_sign_count += 1
                            self.hold_progress = min(self.same_sign_count / self.hold_threshold, 1.0)

                            if self.same_sign_count >= self.hold_threshold:
                                self.add_letter(final_prediction)
                                self.same_sign_count = 0
                                self.hold_progress = 0
                        else:
                            self.last_sign = final_prediction
                            self.same_sign_count = 0
                            self.hold_progress = 0
        else:
            self.current_sign = ""
            self.confidence = 0
            self.hold_progress = 0

            self.frames_without_hand += 1

            if self.frames_without_hand >= self.word_break_threshold:
                if self.current_word:
                    self.finish_word()
                self.frames_without_hand = 0

            self.frame_buffer.clear()
            self.prediction_buffer.clear()
            self.last_sign = None
            self.same_sign_count = 0

        # Update GUI
        self.gui.update_camera(frame)
        self.gui.update_info(
            self.current_sign,
            self.confidence,
            self.hold_progress,
            self.current_word,
            self.suggestions,
            self.current_text,
            self.stats
        )

        # Schedule next frame
        self.gui.root.after(33, self.update_frame)  # ~30 FPS

    def run(self):
        """Run the application"""
        print("Starting GUI...")
        self.gui.root.after(100, self.update_frame)
        self.gui.root.mainloop()


if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run()