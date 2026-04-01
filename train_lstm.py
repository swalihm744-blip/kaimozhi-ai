import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional,
    LayerNormalization, GlobalAveragePooling1D, Multiply, Activation,
    Conv1D, MaxPooling1D
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedShuffleSplit


DATA_PATH  = "MP_Data"
ACTION_DIRS = [
    d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))
]
ACTIONS         = np.array(sorted(ACTION_DIRS))
SEQUENCE_LENGTH = 30
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
EXPECTED_FEATURE_LEN = 126   # 2 hands * 21 landmarks * 3 (x,y,z)
# Number of augmented copies per real sample.
# Increase if you have very few recordings per class.
AUGMENT_COPIES  = 3


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Make landmarks position/scale invariant by translating relative to
    the wrist of whichever hand is present and normalizing by hand span.

    keypoints: flat array of length 126 = [lh(63), rh(63)]
    Each hand block: 21 landmarks * 3 = 63 values, wrist is landmark 0.
    """
    out = keypoints.copy().reshape(2, 21, 3)   # (hand, landmark, xyz)
    for h in range(2):
        hand  = out[h]                          # (21, 3)
        wrist = hand[0].copy()
        # Only normalize if this hand was actually detected (non-zero)
        if not np.allclose(hand, 0):
            hand -= wrist                       # translate so wrist is origin
            scale = np.linalg.norm(hand[9] - hand[0]) + 1e-6   # middle-finger MCP
            hand /= scale
            out[h] = hand
    return out.flatten()


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Augmentation applied to a (T, 126) sequence to improve generalisation:
    - Gaussian coordinate jitter
    - Random horizontal flip (mirror left/right hand)
    - Random time-stretch / temporal jitter
    - Random dropout of a hand (simulates partial occlusion)
    """
    seq = seq.copy()

    # 1. Gaussian noise on coordinates
    seq += np.random.normal(0, 0.007, seq.shape)

    # 2. 50 % chance: mirror x-axis and swap left↔right hand blocks
    if np.random.random() < 0.5:
        r = seq.reshape(seq.shape[0], 2, 21, 3)
        r[:, :, :, 0] *= -1          # flip x
        r = r[:, ::-1, :, :]         # swap hands
        seq = r.reshape(seq.shape)

    # 3. Random temporal jitter: shift sequence start by ±2 frames
    shift = np.random.randint(-2, 3)
    if shift > 0:
        seq = np.concatenate([seq[shift:], np.zeros((shift, seq.shape[1]), dtype=seq.dtype)])
    elif shift < 0:
        seq = np.concatenate([np.zeros((-shift, seq.shape[1]), dtype=seq.dtype), seq[:shift]])

    # 4. 20 % chance: zero out one hand to simulate occlusion
    if np.random.random() < 0.2:
        hand_to_drop = np.random.randint(0, 2)   # 0=left, 1=right
        start = hand_to_drop * 63
        seq[:, start:start + 63] = 0.0

    return seq


def load_sequences(
    data_path: str         = DATA_PATH,
    actions: np.ndarray    = ACTIONS,
    sequence_length: int   = SEQUENCE_LENGTH,
    expected_feature_len: int = EXPECTED_FEATURE_LEN,
    augment_copies: int    = AUGMENT_COPIES,
):
    """Load, normalize, and augment sequences from MP_Data."""
    sequences = []
    labels    = []
    per_class_count = {}

    for label_idx, action in enumerate(actions):
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            print(f"Skipping action '{action}' (folder not found)")
            continue

        sequence_dirs = sorted(
            d for d in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, d))
        )

        count   = 0
        skipped = 0
        for seq_dir in sequence_dirs:
            window   = []
            seq_path = os.path.join(action_path, seq_dir)

            for frame_num in range(sequence_length):
                frame_path = os.path.join(seq_path, f"{frame_num}.npy")
                if not os.path.isfile(frame_path):
                    window = []
                    break

                keypoints = np.load(frame_path)

                if keypoints.shape[0] != expected_feature_len:
                    window   = []
                    skipped += 1
                    break

                keypoints = normalize_keypoints(keypoints)
                window.append(keypoints)

            if len(window) == sequence_length:
                seq_arr = np.array(window, dtype=np.float32)
                sequences.append(seq_arr)
                labels.append(label_idx)
                count += 1

                for _ in range(augment_copies):
                    sequences.append(augment_sequence(seq_arr).astype(np.float32))
                    labels.append(label_idx)

        per_class_count[action] = count
        if skipped:
            print(f"  [{action}] skipped {skipped} seq (wrong feature length)")

    print("\nPer-class sequence counts (before augmentation):")
    for action, cnt in per_class_count.items():
        status = "✓" if cnt > 0 else "✗ NO DATA"
        print(f"  {action:35s}: {cnt:3d}  {status}")
    print()

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels,    dtype=np.int32)
    return X, y


def attention_block(x):
    """Simple temporal self-attention: learns which frames to focus on."""
    score    = Dense(1, activation="tanh")(x)    # (batch, T, 1)
    score    = Activation("softmax")(score)       # softmax over time-axis
    attended = Multiply()([x, score])             # weighted frame features
    return GlobalAveragePooling1D()(attended)     # (batch, units)


def build_model(input_shape, num_classes: int):
    """
    Conv1D feature extractor  → stacked BiLSTM → temporal attention → classifier.

    Conv1D first extracts local motion patterns efficiently,
    BiLSTM learns long-range temporal dependencies from both directions,
    attention focuses on the most discriminative time-steps.
    """
    inp = Input(shape=input_shape)

    # Conv1D front-end: extract local motion patterns
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)            # (batch, T/2, 64)
    x = Dropout(0.2)(x)

    # BiLSTM stack
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    # Temporal attention
    x = attention_block(x)

    # Classifier head
    x   = Dense(128, activation="relu")(x)
    x   = Dropout(0.3)(x)
    x   = Dense(64,  activation="relu")(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)

    # Label smoothing prevents overconfident predictions → better generalisation
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    return model


def train():
    print("Loading data from MP_Data...")
    X, y = load_sequences()

    if X.size == 0:
        raise RuntimeError(
            f"No valid sequences found in '{DATA_PATH}'. "
            "Re-collect data with data_collection.py."
        )

    num_samples = X.shape[0]
    num_classes = len(ACTIONS)
    print(f"Total sequences (with augmentation): {num_samples}")
    print(f"Sequence shape: {X.shape[1:]}")
    print(f"Classes: {num_classes}")

    y_cat = to_categorical(y, num_classes=num_classes)

    # Stratified split — ensures every class is in the val set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(X, y))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_cat[train_idx], y_cat[val_idx]

    print(f"Train sequences: {X_train.shape[0]}")
    print(f"Validation sequences: {X_val.shape[0]}")

    input_shape = (X.shape[1], X.shape[2])
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            "lstm_sign_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    model.save("lstm_sign_model_final.keras")
    np.save("actions.npy", ACTIONS)

    print("\nTraining completed.")
    print("Best model saved as 'lstm_sign_model.keras'.")
    print("Final model saved as 'lstm_sign_model_final.keras'.")
    print("Action labels saved as 'actions.npy'.")

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc * 100:.1f}%")

    return history


if __name__ == "__main__":
    train()
