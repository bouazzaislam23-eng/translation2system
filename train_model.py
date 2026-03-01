"""
Train a simple sign language classifier
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import os
import glob


def load_all_data(data_dir='E:/translation_system/utils/data/collected'):
    """Load all collected data files"""
    all_samples = []
    all_labels = []

    pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))

    if not pkl_files:
        print(f"⚠ No data files found in {data_dir}")
        return None, None

    print(f"Found {len(pkl_files)} data files")

    for filepath in pkl_files:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        for sample in data:
            label = sample['label']
            sequence = sample['sequence']

            # Flatten sequence landmarks into a single feature vector
            features = []
            for frame in sequence:
                if frame['landmarks']:
                    # Use first hand only for simplicity
                    landmarks = frame['landmarks'][0].flatten()
                    features.extend(landmarks)

            if features:
                all_samples.append(features)
                all_labels.append(label)

    return all_samples, all_labels


def train_classifier():
    """Train a simple classifier"""
    print("=" * 60)
    print("TRAINING SIGN LANGUAGE CLASSIFIER")
    print("=" * 60)

    # Load data
    X, y = load_all_data()

    if X is None or len(X) == 0:
        print("\n⚠ No training data available!")
        print("Please run data_collector.py first to collect training data.")
        return

    # Pad sequences to same length
    max_len = max(len(x) for x in X)
    X_padded = []
    for x in X:
        padded = np.pad(x, (0, max_len - len(x)), mode='constant')
        X_padded.append(padded)

    X = np.array(X_padded)
    y = np.array(y)

    print(f"\n✓ Loaded {len(X)} samples")
    print(f"  Classes: {set(y)}")
    print(f"  Feature dimension: {X.shape[1]}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Train model
    print("\n→ Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✓ Training complete!")
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_path = 'models/sign_classifier.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': clf,
            'max_len': max_len,
            'classes': list(set(y))
        }, f)

    print(f"\n✓ Model saved to {model_path}")


if __name__ == "__main__":
    train_classifier()