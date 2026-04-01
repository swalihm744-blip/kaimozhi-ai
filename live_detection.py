import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import tensorflow as tf
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH        = "lstm_sign_model.keras"
ACTIONS_PATH      = "actions.npy"
SEQUENCE_LENGTH   = 30
CONF_THRESHOLD    = 0.70    # minimum confidence to accept a prediction
PREDICTION_STABILITY = 4   # consecutive agreeing preds required
# Predict every N frames (reduces CPU load without sacrificing much accuracy)
PREDICT_EVERY_N   = 2


# ---------------------------------------------------------------------------
# Load model and labels
# ---------------------------------------------------------------------------
model   = load_model(MODEL_PATH)
actions = np.load(ACTIONS_PATH, allow_pickle=True)

# Warm-up inference so the first real call isn't slow
_dummy = np.zeros((1, SEQUENCE_LENGTH, int(model.input_shape[-1])), dtype=np.float32)
_ = model(_dummy, training=False)

# Convert to a concrete function for faster repeated calls (avoids Python overhead)
@tf.function(reduce_retracing=True)
def fast_predict(x):
    return model(x, training=False)


mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Helpers — must match train_lstm.py exactly
# ---------------------------------------------------------------------------

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Make landmarks position/scale invariant by translating relative to the
    wrist of each hand and normalizing by the middle-finger MCP distance.
    Must match the normalization used during training.
    """
    out = keypoints.copy().reshape(2, 21, 3)
    for h in range(2):
        hand = out[h]
        wrist = hand[0].copy()
        if not np.allclose(hand, 0):
            hand -= wrist
            scale = np.linalg.norm(hand[9] - hand[0]) + 1e-6
            hand /= scale
            out[h] = hand
    return out.flatten()


def extract_keypoints(results) -> np.ndarray:
    """
    Extract a 126-dim vector [left_hand(63), right_hand(63)] from a
    MediaPipe result. Zeros are used for absent hands.
    """
    lh = np.zeros(21 * 3, dtype=np.float32)
    rh = np.zeros(21 * 3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label  = results.multi_handedness[i].classification[0].label
            coords = (
                np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                .flatten()
                .astype(np.float32)
            )
            if label == "Left":
                lh = coords
            else:
                rh = coords

    raw = np.concatenate([lh, rh])
    return normalize_keypoints(raw)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)
    # Lower resolution → faster MediaPipe processing → lower lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Request a higher camera FPS if supported
    cap.set(cv2.CAP_PROP_FPS, 60)

    sequence     = deque(maxlen=SEQUENCE_LENGTH)
    recent_preds = deque(maxlen=PREDICTION_STABILITY)

    current_action = ""
    current_prob   = 0.0
    frame_count    = 0

    with mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        max_num_hands=2,
        model_complexity=0,    # lighter model → much faster tracking
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            frame  = cv2.flip(frame, 1)
            image  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # Only run inference every PREDICT_EVERY_N frames to reduce CPU load
            if len(sequence) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY_N == 0:
                input_seq = tf.constant(
                    np.expand_dims(np.array(sequence), axis=0), dtype=tf.float32
                )
                probs    = fast_predict(input_seq)[0].numpy()
                max_idx  = int(np.argmax(probs))
                max_prob = float(probs[max_idx])

                if max_prob > CONF_THRESHOLD:
                    recent_preds.append(max_idx)
                else:
                    recent_preds.append(-1)

                # Accept label only when consecutive predictions agree
                if len(recent_preds) == PREDICTION_STABILITY:
                    most_common, count = Counter(recent_preds).most_common(1)[0]
                    if most_common != -1 and count >= PREDICTION_STABILITY - 1:
                        current_action = str(actions[most_common])
                        current_prob   = float(probs[most_common])
                    elif most_common == -1 and count >= PREDICTION_STABILITY - 1:
                        current_action = ""
                        current_prob   = 0.0

            # ---------------------------------------------------------------
            # HUD
            # ---------------------------------------------------------------
            h, w      = image.shape[:2]
            overlay   = image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            label_text = current_action if current_action else "..."
            cv2.putText(
                image, f"Sign: {label_text}",
                (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (0, 255, 150) if current_action else (180, 180, 180),
                2, cv2.LINE_AA,
            )

            if current_action:
                bar_w = int((w - 40) * current_prob)
                cv2.rectangle(image, (20, 70), (w - 20, 82), (60, 60, 60), -1)
                cv2.rectangle(image, (20, 70), (20 + bar_w, 82), (0, 220, 100), -1)
                cv2.putText(
                    image, f"{current_prob * 100:.0f}%",
                    (w - 70, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            buf_pct = int(len(sequence) / SEQUENCE_LENGTH * 100)
            cv2.putText(
                image, f"Buffer: {buf_pct}%",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA,
            )
            cv2.putText(
                image, "Q: quit",
                (w - 100, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("KaiMozhi Live Detection", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
