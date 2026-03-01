import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from models.hand_detector import HandDetector


class DataCollector:
    """Collect labeled sign language data"""

    def __init__(self, save_dir='data/collected'):
        self.detector = HandDetector()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.current_label = None
        self.samples = []

    def collect(self, label, num_samples=100, frames_per_sample=30):
        """
        Collect samples for a specific sign

        Args:
            label: Name of the sign (e.g., 'hello', 'thanks')
            num_samples: Number of sequences to collect
            frames_per_sample: Number of frames per sequence
        """
        self.current_label = label
        cap = cv2.VideoCapture(0)

        print(f"\n{'=' * 50}")
        print(f"Collecting data for sign: '{label}'")
        print(f"{'=' * 50}")
        print("Instructions:")
        print("  - Press 's' to start recording a sequence")
        print("  - Perform the sign while recording")
        print("  - Press 'q' to quit and save")
        print(f"  - Target: {num_samples} sequences x {frames_per_sample} frames each")

        collecting = False
        sample_count = 0
        frame_count = 0
        current_sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, results = self.detector.find_hands(frame, draw=True)

            # Collect data during recording
            if collecting:
                landmarks = self.detector.get_landmarks(results)
                handedness = self.detector.get_handedness(results)

                if landmarks:  # Only save if hand detected
                    current_sequence.append({
                        'landmarks': landmarks,
                        'handedness': handedness,
                        'frame_num': frame_count
                    })
                    frame_count += 1

                    # Check if sequence is complete
                    if frame_count >= frames_per_sample:
                        self.samples.append({
                            'label': label,
                            'sequence': current_sequence,
                            'timestamp': datetime.now()
                        })
                        sample_count += 1
                        print(f"✓ Sequence {sample_count}/{num_samples} recorded ({len(current_sequence)} frames)")

                        # Reset for next sequence
                        collecting = False
                        frame_count = 0
                        current_sequence = []

                        if sample_count >= num_samples:
                            print(f"\n✓ Collected all {num_samples} sequences for '{label}'!")
                            break

            # Display status
            if collecting:
                status = f"RECORDING: {frame_count}/{frames_per_sample} frames"
                color = (0, 0, 255)  # Red when recording

                # Visual feedback - pulsing circle
                radius = 20 + int(10 * np.sin(frame_count * 0.5))
                cv2.circle(frame, (frame.shape[1] - 40, 40), radius, (0, 0, 255), -1)
            else:
                status = f"Sequences: {sample_count}/{num_samples} | Press 's' to record"
                color = (0, 255, 0)  # Green when ready

            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not collecting:
                collecting = True
                frame_count = 0
                current_sequence = []
                print(f"\n→ Recording sequence {sample_count + 1}...")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_data()

    def save_data(self):
        """Save collected samples to file"""
        if not self.samples:
            print("⚠ No data to save")
            return

        filename = f"{self.save_dir}/{self.current_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self.samples, f)

        print(f"\n✓ Saved {len(self.samples)} sequences to:")
        print(f"  {filename}")

        # Calculate statistics
        total_frames = sum(len(s['sequence']) for s in self.samples)
        avg_frames = total_frames / len(self.samples) if self.samples else 0
        print(f"  Total frames: {total_frames}")
        print(f"  Avg frames per sequence: {avg_frames:.1f}")

        self.samples = []

    def close(self):
        """Cleanup resources"""
        self.detector.close()


if __name__ == "__main__":
    collector = DataCollector()

    # Define signs to collect
    signs_to_collect = [
        ('A', 15, 30), # 5 sequences, 30 frames each
        ('B', 15, 30),
        ('C', 15, 30),
        ('D', 15, 30),
        ('E', 15, 30),
        ('F', 15, 30),
        ('G', 15, 30),
        ('H', 15, 30),
        ('I', 15, 30),
        ('J', 15, 30),
        ('K', 15, 30),
        ('L', 15, 30),
        ('M', 15, 30),
        ('N', 15, 30),
        ('O', 15, 30),
        ('P', 15, 30),
        ('Q', 15, 30),
        ('R', 15, 30),
        ('S', 15, 30),
        ('T', 15, 30),
        ('U', 15, 30),
        ('V', 15, 30),
        ('W', 15, 30),
        ('X', 15, 30),
        ('Y', 15, 30),
        ('Z', 15, 30)
    ]

    print("=" * 60)
    print("SIGN LANGUAGE DATA COLLECTION")
    print("=" * 60)
    print(f"Will collect {len(signs_to_collect)} different signs")
    print()

    try:
        for sign, num_samples, frames in signs_to_collect:
            collector.collect(sign, num_samples=num_samples, frames_per_sample=frames)
            print("\nPress Enter to continue to next sign...")
            input()
    except KeyboardInterrupt:
        print("\n\n⚠ Collection interrupted by user")
    finally:
        collector.close()
        print("\n✓ Data collection complete!")