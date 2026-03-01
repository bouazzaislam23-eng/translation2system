"""
Train a CNN sign language classifier
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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

            # Build sequence of landmarks (don't flatten yet - CNN needs sequences)
            seq_landmarks = []
            for frame in sequence:
                if frame['landmarks']:
                    # Use first hand only
                    landmarks = frame['landmarks'][0]  # Shape: (21, 3)
                    seq_landmarks.append(landmarks)
                else:
                    # No hand detected - add zeros
                    seq_landmarks.append(np.zeros((21, 3)))

            if seq_landmarks:
                all_samples.append(np.array(seq_landmarks))
                all_labels.append(label)

    return all_samples, all_labels


def pad_sequences(sequences):
    """Pad all sequences to same length"""
    max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros at the end
            padding = np.zeros((max_len - len(seq), 21, 3))
            padded_seq = np.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    
    return np.array(padded), max_len


def create_cnn_model(input_shape, num_classes):
    """Create CNN model"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Reshape for Conv1D
        layers.Reshape((input_shape[0], input_shape[1] * input_shape[2])),
        
        # First conv block
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Second conv block
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Third conv block
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_classifier():
    """Train CNN classifier"""
    print("=" * 60)
    print("TRAINING CNN SIGN LANGUAGE CLASSIFIER")
    print("=" * 60)

    # Load data
    X, y = load_all_data()

    if X is None or len(X) == 0:
        print("\n⚠ No training data available!")
        print("Please run data_collector.py first to collect training data.")
        return

    # Pad sequences to same length
    print("\n→ Padding sequences...")
    X, max_len = pad_sequences(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\n✓ Loaded {len(X)} samples")
    print(f"  Classes: {label_encoder.classes_}")
    print(f"  Sequence shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"\n  Train set: {len(X_train)} samples")
    print(f"  Val set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")

    # Create model
    print("\n→ Creating CNN model...")
    input_shape = X_train.shape[1:]  # (timesteps, landmarks, coords)
    num_classes = len(label_encoder.classes_)
    
    model = create_cnn_model(input_shape, num_classes)
    
    print("\nModel architecture:")
    model.summary()

    # Train model
    print("\n→ Training CNN (this will take 2-5 minutes)...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
        ],
        verbose=1
    )

    # Evaluate
    print("\n→ Evaluating model...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✓ Training complete!")
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model
    model_path = 'models/sign_classifier_cnn.keras'
    metadata_path = 'models/sign_classifier_cnn_metadata.pkl'
    
    os.makedirs('models', exist_ok=True)
    
    model.save(model_path)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'max_len': max_len,
            'classes': label_encoder.classes_.tolist(),
            'accuracy': accuracy
        }, f)

    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    train_classifier()
