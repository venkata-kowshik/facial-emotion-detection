import cv2
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def process_webcam_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Webcam not accessible")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logging.info("📷 Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("⚠ Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))

        for (x, y, w, h) in faces:
            logging.info(f"🧠 Face detected at X:{x}, Y:{y}, W:{w}, H:{h}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("🛑 Quitting webcam session")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam_data()
