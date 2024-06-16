import cv2
import mediapipe as mp
import pyautogui
from time import sleep

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
four_fingers_up = -1

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

def is_finger_up(landmarks, tip_idx, pip_idx):
    """
    Determines if a specific finger is up.
    
    Parameters:
    landmarks (list): List of hand landmarks.
    tip_idx (int): Index of the fingertip landmark.
    pip_idx (int): Index of the PIP (proximal interphalangeal) joint landmark.
    
    Returns:
    bool: True if the finger is up, False otherwise.
    """
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def count_fingers_up(hand_landmarks):
    """
    Counts the number of fingers that are up.
    
    Parameters:
    hand_landmarks (NormalizedLandmarkList): Hand landmarks.
    
    Returns:
    int: Number of fingers that are up.
    """
    landmarks = hand_landmarks.landmark
    fingers_up = [
        is_finger_up(landmarks, 8, 6),   # Index finger
        is_finger_up(landmarks, 12, 10), # Middle finger
        is_finger_up(landmarks, 16, 14), # Ring finger
        is_finger_up(landmarks, 20, 18)  # Pinky finger
    ]
    return sum(fingers_up)

while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        break  # Exit the loop if the frame was not captured successfully
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a later selfie-view display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    result = hands.process(rgb_frame)  # Process the frame to detect hands
    
    gesture_detected = None  # Variable to store the detected gesture
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Count the number of fingers that are up
            fingers_up = count_fingers_up(hand_landmarks)
            
            # Determine the gesture based on the number of fingers up
            if fingers_up == 1:
                gesture_detected = 'One Finger Up'
            elif fingers_up == 2:
                gesture_detected = 'Two Fingers Up'
            elif fingers_up == 3:
                gesture_detected = 'Three Fingers Up'
            elif fingers_up == 4:
                gesture_detected = 'Four Fingers Up'
                four_fingers_up *= -1  # Toggle the four_fingers_up variable
                sleep(0.3)  # Add a small delay to avoid rapid toggling
    
    # Display the detected gesture on the frame
    if gesture_detected:
        cv2.putText(frame, gesture_detected, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the four fingers mode status on the frame
    cv2.putText(frame, f'Four Fingers Mode: {"ON" if four_fingers_up > 0 else "OFF"}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)  # Show the frame with the overlay

    # Trigger actions based on the detected gesture and the four fingers mode
    if four_fingers_up < 0:
        if gesture_detected == 'One Finger Up':
            pyautogui.press("volumeup")  # Increase volume
        elif gesture_detected == 'Two Fingers Up':
            pyautogui.press("volumedown")  # Decrease volume
        elif gesture_detected == 'Three Fingers Up':
            pyautogui.press("playpause")  # Play/pause media
            sleep(1)  # Add a delay to avoid rapid toggling

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
