# 🐨 Drowsy Buddy — Your Friendly Alert Companion

Drowsy Buddy is a **real-time drowsiness detection web application** designed to help users stay safe and healthy by monitoring alertness through webcam video.  
If signs of drowsiness are detected continuously, the app **responds kindly**:
- Lets you **take a short rest** with a timer + alarm  
- OR helps you **stay awake** with quick *fun cognitive games* 🎮  
- And then checks if you are awake again 🌞  

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🎥 Live Webcam Monitoring | Detects alert vs. drowsy states in real-time |
| 🧠 CNN + LSTM Model | Trained on driver drowsiness dataset |
| ⚖️ Sustained Detection Logic | Prevents false alarms (ignores blinks) |
| 😴 Rest Mode with Alarm | Choose rest duration; app wakes you up gently |
| 🎮 Wake-Up Games | Reaction & memory mini-games to refresh attention |
| 🐨 Cute Companion UI | Friendly, soft interface instead of harsh warnings |
| 🌐 Runs in Browser | No additional app needed — works via Flask |

---

## 🏗️ System Architecture

Webcam Feed → Preprocessing → CNN Feature Extraction →
LSTM Sequence Learning → Prediction → UI Action (Rest/Game/Continue)

yaml
Copy code

- Model input: 10-frame sequences (145×145 RGB)
- Output: Alert (0) or Drowsy (1)
- Smooth prediction buffer + timed drowsiness threshold

---

## 📦 Installation

### 1) Clone the Repository
```bash
git clone https://github.com/<your-username>/drowsiness-web-app.git
cd drowsiness-web-app
2) Create Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
3) Install Dependencies
bash
Copy code
pip install -r requirements.txt
4) Run the Web App
bash
Copy code
python3 app.py
Now open the link shown in terminal, usually:

cpp
Copy code
http://127.0.0.1:5000
🤖 Model
The app uses:

bash
Copy code
model/my_model.h5   ← REQUIRED
No training data is needed to run the app.

If you want to retrain:

bash
Copy code
python3 train_model.py --use_kaggle --kaggle_slug ismailnasri20/driver-drowsiness-dataset-ddd
🎮 Mini Games
Game	Goal	Skill Used
⚡ Reaction Tap	Tap when screen turns green	Attention & timing
🎨 Color Memory	Repeat color sequences	Short-term working memory

Both games help reset alertness naturally without caffeine ☕.

🐨 Soft Companion UI Preview
mathematica
Copy code
[ Welcome Page ]
   "Hi! I'm your friendly Drowsy Buddy 🐨"
        → Start Detection

[ Detection Mode ]
   Live webcam + smooth alert/drowsy status

[ Drowsiness Detected ]
   "Would you like to rest or perk up?"

 → Rest Mode (timer + alarm)
 → Wake-Up Games 🎮