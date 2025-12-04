from deepface import DeepFace
import cv2
import os

img_path = r"C:\Users\BANKA ANANTHA KUMAR\StressDetector\your_image.jpg"

if not os.path.exists(img_path):
    print(f"❌ Image file not found: {img_path}")
else:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("❌ Failed to load image. Check file format or permissions.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            result = DeepFace.analyze(
                img_path=img_rgb,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                model_name='VGG-Face'
            )

            if isinstance(result, dict):
                emotions = result.get('emotion', {})
                dominant = result.get('dominant_emotion', 'unknown')
            elif isinstance(result, list) and result:
                emotions = result[0].get('emotion', {})
                dominant = result[0].get('dominant_emotion', 'unknown')
            else:
                emotions = {}
                dominant = 'unknown'

            print(f"\n🧠 Dominant Emotion: {dominant}")
            for emotion, score in emotions.items():
                print(f"{emotion}: {score:.2f}%")

        except Exception as e:
            print(f"⚠️ DeepFace analysis failed: {e}")
