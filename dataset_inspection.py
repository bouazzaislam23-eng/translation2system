import pickle
import numpy as np
import os


def inspect_dataset():
    data_dir = 'E:/translation_system/utils/data/collected'

    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return

    pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    if not pkl_files:
        print(f"❌ No .pkl files found in {data_dir}")
        return

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    total_sequences = 0
    total_frames = 0

    for filename in pkl_files:
        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            print(f"\n📁 File: {filename}")
            print(f"   Label: {data[0]['label']}")
            print(f"   Sequences: {len(data)}")

            for i, sample in enumerate(data):
                seq_len = len(sample['sequence'])
                print(f"   Sequence {i + 1}: {seq_len} frames")

                # Show first frame details
                if seq_len > 0:
                    first_frame = sample['sequence'][0]
                    landmarks = first_frame['landmarks']

                    # Handle both list and numpy array
                    if isinstance(landmarks, list) and len(landmarks) > 0:
                        landmarks = landmarks[0]  # Get first hand if it's a list

                    # Convert to numpy array if it's a list
                    if not isinstance(landmarks, np.ndarray):
                        landmarks = np.array(landmarks)

                    print(f"      Hand: {first_frame.get('handedness', 'Unknown')}")
                    print(f"      Landmarks shape: {landmarks.shape}")

                    if len(landmarks) >= 9:
                        print(f"      First landmark (wrist): {landmarks[0][:3]}")  # First 3 values
                        print(f"      Thumb tip: {landmarks[4][:3] if len(landmarks) > 4 else 'N/A'}")
                        print(f"      Index tip: {landmarks[8][:3] if len(landmarks) > 8 else 'N/A'}")

            total_sequences += len(data)
            total_frames += sum(len(s['sequence']) for s in data)

        except Exception as e:
            print(f"\n❌ Error reading {filename}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(pkl_files)}")
    print(f"Total sequences: {total_sequences}")
    print(f"Total frames: {total_frames}")
    if total_frames > 0:
        print(f"Total data points: {total_frames * 21 * 3:,}")  # frames × landmarks × coordinates
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()