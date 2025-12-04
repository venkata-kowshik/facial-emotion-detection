import cv2
import random
import threading
import os
import time
import webbrowser
import logging
from collections import deque
from deepface import DeepFace

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class Config:
    STRESS_EMOTIONS = ["angry", "fear", "sad", "disgust"]
    RELAX_EMOTIONS = ["happy", "neutral"]
    LOCAL_SONGS = ["calm1.mp3", "calm2.mp3", "calm3.mp3"]
    HAPPY_SPOTIFY_LINKS = [
        "https://open.spotify.com/playlist/37i9dQZF1DZ06evNZVVBPG?si=erPB-UqbQLam9ZsdRTpBzA",
        "https://open.spotify.com/playlist/1omqR4qHb4SnSn61367Dch?si=9TkUXmWNRACX_nx0A18-fw",
        "https://open.spotify.com/playlist/5AB5tIUaoGyIsa5U2vY8Z9?si=vZ5hQPQZSIKvSW6IITF4HA",
        "https://open.spotify.com/playlist/105A9IjiVCv6iWft9qjxe2?si=5SLrmzYfTg-gxEx0pWKCWA",
        "https://open.spotify.com/playlist/5QOrHPIzTFh80WhHmbOcCp?si=fDelEESSSgS-ePH5NF061w",
        "https://open.spotify.com/playlist/37i9dQZF1DZ06evNZVVBPG",
    ]
    STRESS_TIPS = [
        "Take a deep breath and count to 10.",
        "Stretch your arms and legs for a minute.",
        "Close your eyes and visualize a peaceful place.",
        "Drink a glass of water slowly.",
        "Step away from the screen for a short walk.",
    ]
    ANALYSIS_COOLDOWN = 5

def play_local_song():
    song = random.choice(Config.LOCAL_SONGS)
    if os.path.exists(song):
        try:
            from playsound import playsound
            threading.Thread(target=playsound, args=(song,), daemon=True).start()
            logging.info(f"🎵 Playing: {song}")
        except:
            logging.warning("⚠ Audio playback not available")
    else:
        logging.warning(f"⚠ Song not found: {song}")

def play_spotify_song():
    link = random.choice(Config.HAPPY_SPOTIFY_LINKS)
    webbrowser.open(link)
    logging.info(f"🎧 Opening Spotify")

def play_guessing_game():
    number = random.randint(1, 10)
    attempts = 3
    print("\n🎲 Guess a number (1-10):")
    while attempts > 0:
        try:
            guess = int(input(f"Attempts left {attempts}: "))
            if guess == number:
                print("✅ Correct!")
                return
            print("🔻 Too low!" if guess < number else "🔺 Too high!")
            attempts -= 1
        except ValueError:
            print("⚠ Enter a valid number")
    print(f"❌ The number was {number}")

def offer_relief_menu():
    print("\n🧘 Stress Relief Options:")
    print("1. 💡 Stress tip")
    print("2. 🎮 Guessing game") 
    print("3. 🎵 Spotify song")
    print("4. 🎶 Local song")
    print("5. ❌ Exit")

    while True:
        choice = input("Choose (1-5): ").strip()
        if choice == "1":
            print(f"💡 {random.choice(Config.STRESS_TIPS)}")
        elif choice == "2":
            play_guessing_game()
        elif choice == "3":
            play_spotify_song()
        elif choice == "4":
            play_local_song()
        elif choice == "5":
            print("🌿 Relief session completed\n")
            break
        else:
            print("⚠ Invalid choice")

def detect_tears(emotions):
    """Detect happy and sad tears based on emotion combinations"""
    happy = emotions.get("happy", 0)
    sad = emotions.get("sad", 0)
    total = sum(emotions.values())
    
    if total > 0:
        happy_ratio = happy / total
        sad_ratio = sad / total
        
        if happy_ratio > 0.4 and sad_ratio > 0.3:
            return "🥲 Happy Tears - joy mixed with sadness"
        elif sad_ratio > 0.4 and happy_ratio > 0.3:
            return "😢 Sad Tears - sorrow with a touch of hope"
    
    return None

def analyze_emotions(face_img):
    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False, silent=True)
        
        if isinstance(result, dict):
            return result.get('emotion', {}), result.get('dominant_emotion', 'unknown')
        elif isinstance(result, list) and result:
            return result[0].get('emotion', {}), result[0].get('dominant_emotion', 'unknown')
        return None, None
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return None, None

def detect_emotion():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("❌ Webcam not accessible")
        return

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
                    
                    # Detect tears
                    tears_message = detect_tears(emotions)

                    print(f"\n🧠 Stable Emotion: {stable_emotion}")
                    for emotion, percent in emotions.items():
                        print(f"{emotion}: {percent:.1f}%")
                    
                    if tears_message:
                        print(f"💧 {tears_message}")

                    stress_score = sum(emotions.get(e, 0) for e in Config.STRESS_EMOTIONS)
                    
                    if stress_score > 60:
                        logging.warning(f"⚠ High stress detected!")
                        threading.Thread(target=offer_relief_menu, daemon=True).start()
                    
                    last_detection_time = time.time()

        # Display emotions on camera feed
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

if __name__ == "__main__":
    detect_emotion()