# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:14:21 2023

@author: Pranav
"""
!pip install --upgrade pip

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pickle
from joblib import load
import streamlit.components.v1 as components

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_hands = mp.solutions.hands
FRAME_WINDOW = st.image([])


#model = pickle.load(open('C:/Users/Pranav/Desktop/Gesture_detection/model.pkl', 'rb'))
pickle.load(open('Gesture_detection/model.pkl', 'rb'))


def record_landmarks(specter):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if specter.multi_hand_landmarks:
        for i, every_hand in enumerate(specter.multi_hand_landmarks):
            if i == 0:
                for j, landmark in enumerate(every_hand.landmark):
                    left[j] = [landmark.x, landmark.y, landmark.z]
            elif i == 1:
                for j, landmark in enumerate(every_hand.landmark):
                    right[j] = [landmark.x, landmark.y, landmark.z]

    return np.concatenate((left, right), axis=0)

def main():
    st.title("Hand Gesture Detection")


    perform_analysis = st.button("Click to turn on cam")


    col1, col2 = st.columns(2)

    gesture_placeholder = col2.empty()
    FRAME_WINDOW = col1.image([])





    #CSS
    main_bg_color = 'linear-gradient(to bottom, #F0F2F5, #CBD2D9)'
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: {main_bg_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if perform_analysis:
        cap = cv2.VideoCapture(0)
        hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        while True:
            success, actual_image = cap.read()
            if actual_image is not None:
                actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(actual_image)

            with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                actual_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2RGB)
                actual_image.flags.writeable = False
                results = hands.process(actual_image)
                actual_image.flags.writeable = True
                actual_image = cv2.cvtColor(actual_image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            actual_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style()
                        )

                        landmarks = record_landmarks(results)
                        landmarks = landmarks.astype(np.float32)
                        landmarks = landmarks.ravel().reshape(1, -1)
                        gesture_pred = model.predict(landmarks)

                        #gesture_placeholder.write("Predicted Gesture \n")
                        gesture_placeholder.markdown(
                            f"<div class='gesture-text'> Predicted Gesture :: {gesture_pred[0]}</div>",
                            unsafe_allow_html=True
                        )

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()

if __name__ == '__main__':
    main()
