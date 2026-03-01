"""
Train CNN on Full Alphabet Dataset
Optimized for 29 classes and large data
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
    """Load all collected data"""
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))
    
    if not pkl_files:
        print(f"⚠ No data files found in {data_dir}")
        return None, None
    
    print(f"\nFound {len(pkl_files)} data files")
    
    all_samples = []
    all_labels = []
    class_counts = {}
    
    for filepath in pkl_files:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        for sample in data:
            label = sample['label']
            sequence = sample['sequence']
            
            # Build sequence of landmarks
            seq_landmarks = []

            for frame in sequence:
                landmarks = frame['landmarks']

                if landmarks is None:
                    seq_landmarks.append(np.zeros(63))
                    continue

                arr = np.array(landmarks)

                # Force reshape safely
                if arr.ndim == 2 and arr.shape == (21, 3):
                    arr = arr.flatten()

                arr = arr.flatten()

                # Safety check
                if arr.shape[0] != 63:
                    print("⚠ Invalid landmark shape:", arr.shape)
                    continue

                seq_landmarks.append(arr)

            # Only keep sequences that are clean
            if len(seq_landmarks) == 30:
                all_samples.append(np.array(seq_landmarks))
                all_labels.append(label)
                class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"\n✓ Loaded {len(all_samples)} samples")
    print(f"✓ Classes: {len(class_counts)}")
    
    print("\nSamples per class:")
    for label in sorted(class_counts.keys()):
        print(f"  {label:8s}: {class_counts[label]:4d}")

    print("Final sample shape example:", np.array(all_samples[0]).shape)
    return all_samples, all_labels


def pad_sequences(sequences):
    """Pad all sequences to same length"""
    max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        seq = np.array(seq)
        if len(seq) < max_len:
            feature_dim = seq.shape[1]
            padding = np.zeros((max_len - len(seq), feature_dim))
            padded_seq = np.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    
    return np.array(padded), max_len


def create_large_cnn_model(input_shape, num_classes):
    """
    Create optimized CNN for large alphabet
    Enhanced architecture for 29 classes
    """
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Block 2
        layers.Conv1D(256, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv1D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_full_alphabet():
    """Train on full alphabet"""
    print("\n" + "=" * 70)
    print("FULL ALPHABET CNN TRAINING")
    print("=" * 70)
    
    # Load data
    # Load data
    X, y = load_all_data()

    if X is None or len(X) == 0:
        print("\n⚠ No training data!")
        print("Run convert_full_alphabet.py first.")
        return

    # --------------------------------------------------
    # 🔥 FILTER CLASSES WITH TOO FEW SAMPLES
    # --------------------------------------------------
    from collections import Counter

    class_counts = Counter(y)
    print("\nOriginal class distribution:", class_counts)

    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}

    filtered_X = []
    filtered_y = []

    for sample, label in zip(X, y):
        if label in valid_classes:
            filtered_X.append(sample)
            filtered_y.append(label)

    X = filtered_X
    y = filtered_y

    print("Filtered class distribution:", Counter(y))
    # --------------------------------------------------

    # Pad sequences
    print("\n" + "=" * 70)
    print("PREPROCESSING")
    print("=" * 70)
    print("\nPadding sequences...")
    X, max_len = pad_sequences(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✓ Data shape: {X.shape}")
    print(f"✓ Classes: {len(label_encoder.classes_)}")
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Create model
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    input_shape = X_train.shape[1:]
    num_classes = len(label_encoder.classes_)
    
    model = create_large_cnn_model(input_shape, num_classes)
    
    print(f"\nModel for {num_classes} classes")
    print(f"Input shape: {input_shape}")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'models/best_alphabet_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    print("\nThis will take 10-20 minutes...")
    print("Training CNN on full alphabet...\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Test Accuracy: {accuracy * 100:.2f}%")
    
    print("\nPer-class Performance:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=3
    ))
    
    # Save
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/sign_classifier_cnn.keras'
    metadata_path = 'models/sign_classifier_cnn_metadata.pkl'
    
    model.save(model_path)
    
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'max_len': max_len,
            'classes': label_encoder.classes_.tolist(),
            'accuracy': accuracy,
            'num_classes': num_classes
        }, f)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Metadata saved: {metadata_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Classes trained: {num_classes}")
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test accuracy: {accuracy * 100:.2f}%")
    print(f"\nNext step:")
    print(f"  python inference_smart.py")
    print("=" * 70)


if __name__ == "__main__":
    train_full_alphabet()
