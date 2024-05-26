import cv2
import mediapipe as mp

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the Iron Man hand armor image
ironman_hand = cv2.imread('img7.png', cv2.IMREAD_UNCHANGED)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay img_overlay on top of img at (x, y) and blend using alpha_mask."""
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1_o, y2_o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1_o, x2_o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no overlay is necessary
    if y1 >= y2 or x1 >= x2 or y1_o >= y2_o or x1_o >= x2_o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1_o:y2_o, x1_o:x2_o]

    alpha = alpha_mask[y1_o:y2_o, x1_o:x2_o, 3] / 255.0
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = (alpha * img_overlay_crop[:, :, c] + alpha_inv * img_crop[:, :, c])

    img[y1:y2, x1:x2] = img_crop

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw hand landmarks and overlay the Iron Man hand armor
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of wrist (landmark 0) and middle finger tip (landmark 12)
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            h, w, _ = image.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Calculate the center point of the hand
            center_x = (wrist_x + middle_x) // 2
            center_y = (wrist_y + middle_y) // 2

            # Calculate the position and size for the Iron Man hand armor
            armor_width = abs(middle_x - wrist_x)
            armor_height = abs(middle_y - wrist_y) * 2  # Adjust the height as needed
            armor_x = center_x - armor_width // 2
            armor_y = center_y - armor_height // 2

            # Resize the Iron Man hand armor to fit the hand
            ironman_hand_resized = cv2.resize(ironman_hand, (armor_width, armor_height))

            # Overlay the Iron Man hand armor on the image
            overlay_image_alpha(image, ironman_hand_resized, armor_x, armor_y, ironman_hand_resized)

            # The following line is commented out to hide hand landmarks
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Iron Man Hand', image)

    # Break the loop on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
