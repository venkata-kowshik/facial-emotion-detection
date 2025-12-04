import cv2
from deepface import DeepFace
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def analyze_emotions(face_img):
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            face_rgb,
            actions=['emotion'],
            enforce_detection=False,
            silent=True,
            detector_backend='opencv',
            model_name='VGG-Face'
        )
        
        if isinstance(result, dict):
            return result.get('emotion', {}), result.get('dominant_emotion', 'unknown')
        elif isinstance(result, list) and result:
            return result[0].get('emotion', {}), result[0].get('dominant_emotion', 'unknown')
        return None, None
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None, None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return

    print("📷 Webcam active. Press 'c' to capture and analyze, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Emotion Detector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            emotions, dominant = analyze_emotions(frame)
            if emotions:
                print(f"\n🧠 Dominant Emotion: {dominant}")
                for emotion, score in emotions.items():
                    print(f"{emotion}: {score:.2f}%")
            else:
                print("⚠ No emotions detected.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
