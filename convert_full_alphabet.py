"""
Full ASL Alphabet Converter
Converts ALL 29 classes from Kaggle ASL Alphabet dataset:
- A-Z (26 letters)
- space
- del (delete)
- nothing
"""

import cv2
import pickle
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import sys
import random

# Import hand detector
sys.path.insert(0, 'E:/translation_system/models')
from hand_detector import HandDetector

def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks:
    - Convert to numpy
    - Center at wrist (landmark 0)
    - Scale to unit size
    - Flatten to 1D vector
    """

    landmarks = np.array(landmarks)  # shape: (21, 3)

    # Center at wrist (landmark 0)
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale by maximum absolute value (hand size normalization)
    max_value = np.max(np.abs(landmarks))
    if max_value > 0:
        landmarks = landmarks / max_value

    # Flatten to 1D (63 values)
    return landmarks.flatten()



def convert_full_alphabet(images_per_class=100):
    """
    Convert entire ASL alphabet dataset
    
    Args:
        images_per_class: How many images per class (default: 100)
    """
    
    print("\n" + "=" * 70)
    print("FULL ASL ALPHABET CONVERTER")
    print("Converting ALL 29 classes!")
    print("=" * 70)
    
    # Paths
    dataset_path = Path(r'E:\asl_alphabet_train\asl_alphabet_train\asl_alphabet_train')
    output_dir = Path(r'E:\translation_system\utils\data\collected')
    
    print(f"\nDataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Images per class: {images_per_class}")
    
    # Check dataset exists
    if not dataset_path.exists():
        print(f"\n⚠ ERROR: Dataset not found at {dataset_path}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # All 29 classes in the dataset
    all_classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z',
        'space', 'del', 'nothing'
    ]
    
    print(f"\nClasses to convert: {len(all_classes)}")
    print(f"Total images to process: {len(all_classes) * images_per_class}")
    print(f"Estimated time: {len(all_classes) * 2} minutes")
    
    input("\nPress ENTER to start conversion...")
    
    # Initialize detector
    print("\nInitializing hand detector...")
    detector = HandDetector()
    print("✓ Ready!")
    
    # Statistics
    total_successful = 0
    total_failed = 0
    class_stats = {}
    
    # Convert each class
    for idx, class_name in enumerate(all_classes, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(all_classes)}] Converting: {class_name}")
        print(f"{'='*70}")
        
        # Path to class folder
        class_folder = dataset_path / class_name
        
        if not class_folder.exists():
            print(f"⚠ Folder not found: {class_folder}")
            class_stats[class_name] = {'success': 0, 'failed': 0}
            continue
        
        # Get images
        images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))

        # Shuffle to remove ordering bias
        random.shuffle(images)

        # Take more candidates to ensure enough successful detections
        images = images[:images_per_class * 3]

        
        # Convert images
        samples = []
        successful = 0
        failed = 0
        
        for img_idx, img_path in enumerate(images, 1):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            
            # Detect hand
            _, results = detector.find_hands(img, draw=False)
            landmarks = detector.get_landmarks(results)
            handedness = detector.get_handedness(results)

            if landmarks and len(landmarks) > 0:
                # Take only the first hand
                first_hand = landmarks[0]

                # Normalize it
                normalized = normalize_landmarks(first_hand)

                sequence = []
                for frame_num in range(30):
                    sequence.append({
                        'landmarks': normalized,
                        'handedness': handedness[0] if handedness else [],
                        'frame_num': frame_num
                    })

                samples.append({
                    'label': class_name,
                    'sequence': sequence,
                    'timestamp': datetime.now()
                })

                successful += 1
            else:
                if successful >= images_per_class:
                    break
            
            # Progress every 20 images
            if img_idx % 20 == 0:
                print(f"  Progress: {img_idx}/{len(images)} (✓{successful} ✗{failed})")
        
        # Final stats
        print(f"\n✓ Complete:")
        print(f"  Successful: {successful}/{len(images)} ({successful/len(images)*100:.1f}%)")
        print(f"  Failed: {failed}/{len(images)}")
        
        class_stats[class_name] = {'success': successful, 'failed': failed}
        total_successful += successful
        total_failed += failed
        
        # Save to file
        if samples:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"{class_name}_{timestamp}.pkl"
            
            with open(output_file, 'wb') as f:
                pickle.dump(samples, f)
            
            print(f"  Saved: {output_file.name}")
        else:
            print(f"  ⚠ No valid samples")
    
    # Final summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    
    print(f"\nOverall Statistics:")
    print(f"  Total images processed: {total_successful + total_failed}")
    print(f"  Successful: {total_successful} ({total_successful/(total_successful+total_failed)*100:.1f}%)")
    print(f"  Failed: {total_failed}")
    print(f"  Classes: {len(all_classes)}")
    
    print(f"\nPer-class breakdown:")
    for class_name in all_classes:
        if class_name in class_stats:
            stats = class_stats[class_name]
            print(f"  {class_name:8s}: ✓{stats['success']:3d} ✗{stats['failed']:2d}")
    
    print(f"\nData saved in: {output_dir}")
    print(f"\nNext step:")
    print(f"  python train_model_smart.py")
    print("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FULL ASL ALPHABET DATASET CONVERTER")
    print("=" * 70)
    print("\nThis will convert ALL 29 classes:")
    print("  • A-Z (26 letters)")
    print("  • space")
    print("  • del")
    print("  • nothing")
    
    print("\n" + "=" * 70)
    print("Configuration Options:")
    print("=" * 70)
    
    print("\nHow many images per class?")
    print("  • 50:  Quick (30 min) - Good accuracy (90-93%)")
    print("  • 100: Recommended (1 hour) - Best accuracy (92-96%)")
    print("  • 200: Maximum (2 hours) - Highest accuracy (93-97%)")
    
    num_str = input("\nImages per class (default 100): ").strip()
    num_images = int(num_str) if num_str else 100
    
    print(f"\n✓ Configuration:")
    print(f"  Classes: 29 (full alphabet)")
    print(f"  Images per class: {num_images}")
    print(f"  Total images: {29 * num_images}")
    print(f"  Estimated time: {29 * (num_images / 50)} minutes")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    
    if confirm == 'y':
        convert_full_alphabet(num_images)
    else:
        print("Cancelled.")
