"""
Dataset Inspection Tool
Updated to match current data format (normalized, flattened landmarks)
"""

import pickle
import numpy as np
import os


def inspect_dataset():
    """Inspect collected dataset and show statistics"""
    data_dir = 'E:/translation_system/utils/data/collected'

    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return

    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"❌ No .pkl files found in {data_dir}")
        return

    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    total_sequences = 0
    total_frames = 0
    class_distribution = {}

    for filename in sorted(pkl_files):
        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            if not data:
                print(f"\n⚠ Empty file: {filename}")
                continue

            label = data[0]['label']
            num_sequences = len(data)

            print(f"\n📁 File: {filename}")
            print(f"   Label: {label}")
            print(f"   Sequences: {num_sequences}")

            # Update class distribution
            class_distribution[label] = class_distribution.get(label, 0) + num_sequences

            # Inspect first sequence
            if num_sequences > 0:
                first_sample = data[0]
                sequence = first_sample['sequence']
                seq_len = len(sequence)

                print(f"   Sequence length: {seq_len} frames")

                # Inspect first frame
                if seq_len > 0:
                    first_frame = sequence[0]
                    landmarks = first_frame['landmarks']

                    # Convert to numpy if needed
                    if not isinstance(landmarks, np.ndarray):
                        landmarks = np.array(landmarks)

                    print(f"   Handedness: {first_frame.get('handedness', 'Unknown')}")
                    print(f"   Landmarks shape: {landmarks.shape}")
                    print(f"   Landmarks dtype: {landmarks.dtype}")

                    # Handle flattened format (63 values)
                    if landmarks.shape == (63,):
                        print(f"   Format: Flattened (normalized)")
                        print(
                            f"   First 3 values (wrist x,y,z): [{landmarks[0]:.4f}, {landmarks[1]:.4f}, {landmarks[2]:.4f}]")
                        print(
                            f"   Thumb tip (indices 12-14): [{landmarks[12]:.4f}, {landmarks[13]:.4f}, {landmarks[14]:.4f}]")
                        print(
                            f"   Index tip (indices 24-26): [{landmarks[24]:.4f}, {landmarks[25]:.4f}, {landmarks[26]:.4f}]")
                        print(f"   Value range: [{landmarks.min():.4f}, {landmarks.max():.4f}]")

                    # Handle 2D format (21, 3)
                    elif landmarks.shape == (21, 3):
                        print(f"   Format: 2D array (21 landmarks × 3 coords)")
                        print(f"   Wrist (landmark 0): {landmarks[0]}")
                        print(f"   Thumb tip (landmark 4): {landmarks[4]}")
                        print(f"   Index tip (landmark 8): {landmarks[8]}")

                    # Handle nested list format
                    elif isinstance(landmarks, (list, np.ndarray)) and len(landmarks) > 0:
                        if isinstance(landmarks[0], (list, np.ndarray)):
                            print(f"   Format: Nested array")
                            first_hand = landmarks[0]
                            print(f"   First hand shape: {np.array(first_hand).shape}")
                        else:
                            print(f"   Format: Unknown - {type(landmarks)}")

                    else:
                        print(
                            f"   ⚠ Unknown format: {type(landmarks)}, shape: {landmarks.shape if hasattr(landmarks, 'shape') else 'N/A'}")

            total_sequences += num_sequences
            total_frames += sum(len(s['sequence']) for s in data)

        except Exception as e:
            print(f"\n❌ Error reading {filename}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(pkl_files)}")
    print(f"Total sequences: {total_sequences}")
    print(f"Total frames: {total_frames}")

    if total_frames > 0:
        print(f"Total data points: {total_frames * 63:,} (assuming flattened format)")

    print(f"\nClass Distribution:")
    for label in sorted(class_distribution.keys()):
        count = class_distribution[label]
        percentage = (count / total_sequences * 100) if total_sequences > 0 else 0
        print(f"  {label:10s}: {count:4d} sequences ({percentage:5.1f}%)")

    # Data quality checks
    print(f"\nData Quality:")
    if total_sequences > 0:
        avg_frames = total_frames / total_sequences
        print(f"  Average frames per sequence: {avg_frames:.1f}")

        if avg_frames == 30:
            print(f"  ✓ All sequences have expected 30 frames")
        else:
            print(f"  ⚠ Frame count varies (expected 30)")

    # Class balance check
    if class_distribution:
        counts = list(class_distribution.values())
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        print(f"\nClass Balance:")
        print(f"  Min samples: {min_count}")
        print(f"  Max samples: {max_count}")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio < 2.0:
            print(f"  ✓ Classes are well balanced")
        elif imbalance_ratio < 5.0:
            print(f"  ⚠ Moderate class imbalance")
        else:
            print(f"  ❌ Significant class imbalance")

    print("=" * 70)


def validate_for_training():
    """Check if dataset is ready for training"""
    data_dir = 'E:/translation_system/utils/data/collected'

    print("\n" + "=" * 70)
    print("TRAINING READINESS CHECK")
    print("=" * 70)

    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')] if os.path.exists(data_dir) else []

    checks = {
        'files_exist': len(pkl_files) > 0,
        'min_classes': len(pkl_files) >= 3,
        'enough_data': False,
        'consistent_format': True
    }

    if checks['files_exist']:
        total_sequences = 0
        format_issues = 0

        for filename in pkl_files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                total_sequences += len(data)

                # Check format consistency
                if data and len(data[0]['sequence']) > 0:
                    landmarks = data[0]['sequence'][0]['landmarks']
                    if isinstance(landmarks, np.ndarray):
                        if landmarks.shape != (63,):
                            format_issues += 1

            except:
                format_issues += 1

        checks['enough_data'] = total_sequences >= 60
        checks['consistent_format'] = format_issues == 0

    # Print results
    print("\nChecks:")
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check.replace('_', ' ').title()}")

    all_passed = all(checks.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ READY FOR TRAINING")
        print("Run: python train_full_alphabet.py")
    else:
        print("❌ NOT READY FOR TRAINING")
        if not checks['files_exist']:
            print("→ No data files found")
            print("   Run: python convert_full_alphabet.py")
        elif not checks['min_classes']:
            print("→ Need at least 3 classes")
            print("   Collect more data with convert_full_alphabet.py")
        elif not checks['enough_data']:
            print("→ Need at least 60 total sequences")
            print("   Increase images_per_class in convert_full_alphabet.py")
        elif not checks['consistent_format']:
            print("→ Data format inconsistency detected")
            print("   Re-run convert_full_alphabet.py")

    print("=" * 70)


if __name__ == "__main__":
    inspect_dataset()
    validate_for_training()