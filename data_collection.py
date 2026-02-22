import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Dataset folder
DATA_PATH = 'MP_Data'

# Words to collect
actions = ['Hello', 'Thanks', 'Yes']

# Videos per word (Recommended minimum)
no_sequences = 40

# Frames per video
sequence_length = 30


# Create folder structure automatically
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


# Start Camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
) as hands:

    for action in actions:
        for sequence in range(no_sequences):

            print(f"Collecting -> {action} | Video {sequence}")

            # ⭐ 3 SECOND COUNTDOWN
            for i in range(3,0,-1):
                ret, frame = cap.read()

                cv2.putText(frame,
                            f'Starting in {i}',
                            (200,250),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0,0,255),
                            4)

                cv2.imshow('KaiMozhi Data Collection', frame)
                cv2.waitKey(1000)

            # Capture frames
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                if not ret:
                    break

                # Convert color
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Default empty hands
                lh = np.zeros(21*3)
                rh = np.zeros(21*3)

                # Detect hands
                if results.multi_hand_landmarks:
                    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):

                        label = results.multi_handedness[i].classification[0].label

                        coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        ).flatten()

                        if label == 'Left':
                            lh = coords
                        else:
                            rh = coords

                        mp_draw.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                keypoints = np.concatenate([lh, rh])

                # Save data
                np.save(
                    os.path.join(DATA_PATH, action, str(sequence), str(frame_num)),
                    keypoints
                )

                # Display info
                cv2.putText(image,
                            f'Action: {action}',
                            (10,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                cv2.putText(image,
                            f'Video: {sequence}/{no_sequences}',
                            (10,80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,0),
                            2)

                cv2.imshow('KaiMozhi Data Collection', image)

                # ⭐ PRESS Q TO STOP CAMERA
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
