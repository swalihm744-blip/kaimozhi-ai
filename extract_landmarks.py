import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = '2. Dynamic Sign Gesture'
SAVE_PATH = 'MP_Data'

actions = os.listdir(DATA_PATH)

for action in actions:
    videos = os.listdir(os.path.join(DATA_PATH, action))
    
    for vid_num, video in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(DATA_PATH, action, video))
        
        for frame_num in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            keypoints = np.zeros(21*3)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    ).flatten()
            
            save_dir = os.path.join(SAVE_PATH, action, str(vid_num))
            os.makedirs(save_dir, exist_ok=True)
            
            np.save(os.path.join(save_dir, str(frame_num)), keypoints)
        
        cap.release()

print("Extraction Completed Successfully!")