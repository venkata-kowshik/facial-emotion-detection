import cv2
import time
import threading
import logging
from collections import deque

# Make sure these functions and config are defined elsewhere in your code
# analyze_emotions(), detect_tears(), offer_relief_menu(), Config.STRESS_EMOTIONS, Config.ANALYSIS_COOLDOWN

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("📷 Webcam active. Press 'q' to quit")

last_detection_time = 0
recent_emotions = deque(maxlen=5)
emotion_display = {}
dominant_display = ""
tears_message = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Analyzing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if time.time() - last_detection_time > Config.ANALYSIS_COOLDOWN:
            face_img = frame[y:y+h, x:x+w]
            emotions, dominant_emotion = analyze_emotions(face_img)

            if emotions:
                recent_emotions.append(dominant_emotion)
                stable_emotion = max(set(recent_emotions), key=recent_emotions.count)

                emotion_display = emotions
                dominant_display = stable_emotion
                tears_message = detect_tears(emotions)

                print(f"\n🧠 Stable Emotion: {stable_emotion}")
                for emotion, percent in emotions.items():
                    print(f"{emotion}: {percent:.1f}%")

                if tears_message:
                    print(f"💧 {tears_message}")

                stress_score = sum(emotions.get(e, 0) for e in Config.STRESS_EMOTIONS)
                if stress_score > 60:
                    logging.warning("⚠ High stress detected!")
                    threading.Thread(target=offer_relief_menu, daemon=True).start()

                last_detection_time = time.time()

    y_offset = 30
    for emotion, percent in emotion_display.items():
        text = f"{emotion}: {percent:.1f}%"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25

    if dominant_display:
        cv2.putText(frame, f"Dominant: {dominant_display}", (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if tears_message:
        cv2.putText(frame, tears_message, (10, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()