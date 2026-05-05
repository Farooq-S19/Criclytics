# 🏏 Criclytics - AI-Powered Cricket Analytics Platform

Criclytics is a **web-based cricket analytics platform** built using **Flask and Machine Learning**. It provides deep insights into player performance and predicts match outcomes using intelligent data-driven models.

---



## 🚀 Core Features

### 🔹 1. In-depth Player Analysis

* 🔍 **Player Search** – Select any player from dataset
* 🤖 **AI Score Prediction** – Predicts player performance using RandomForestRegressor
* 📊 **Dynamic Visualizations**

  * Bar chart for recent performance
  * Pie chart for scoring breakdown
  * Radar chart for skill analysis
* 🧠 **Smart Player Summary** – Auto-generated insights

---

### 🔹 2. Team vs Team Prediction

* ⚔️ **Head-to-Head Analysis**
* 📈 **Win Probability (%)**
* ⭐ **Key Players Identification**
* 📊 **Team Statistics Overview**
* 🧾 **AI-Based Match Reasoning**

---

## 🛠️ Technology Stack

| Category         | Technologies Used            |
| ---------------- | ---------------------------- |
| Backend          | Python, Flask                |
| Machine Learning | Scikit-learn, Pandas, Joblib |
| Frontend         | HTML, CSS, JavaScript        |
| Visualization    | Chart.js                     |
| Data Storage     | CSV (players_data.csv)       |

---

## 📁 Project Structure

```bash
criclytics/
│── app.py
│── model.py
│── requirements.txt
│
├── static/
│   ├── css/
│   ├── js/
│   └── data/
│       └── players_data.csv
│
├── templates/
│   └── index.html
│
├── models/
│   ├── player_score_model.pkl
│   └── feature_scaler.pkl
```

---

## ⚙️ Setup & Installation

Follow these steps to run the project locally 👇

---

### ✅ Prerequisites

* Python 3.7+
* pip

---

### 🔽 1. Clone Repository

```bash
git clone https://github.com/your-username/criclytics-flask.git
cd criclytics-flask
```

---

### 🧪 2. Create Virtual Environment

#### ▶️ macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### ▶️ Windows

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 📊 4. Prepare Data & Train Model (Important)

#### 📁 Add Dataset

* Place `players_data.csv` inside:

```
static/data/
```

#### 🤖 Train Model

```bash
python model.py
```

👉 This generates:

* `player_score_model.pkl`
* `feature_scaler.pkl`

(saved in `models/` folder)

---

### ▶️ 5. Run Flask App

```bash
flask run
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## ✨ Highlights

* ⚡ Fast and lightweight Flask backend
* 🧠 AI-powered predictions
* 📊 Interactive charts using Chart.js
* 🎯 Clean and user-friendly UI

---

## 📌 Future Improvements

* Live match data integration
* Player comparison feature
* Deployment on cloud (Render / AWS)
* User authentication system

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

* Fork the repo
* Create a new branch
* Submit a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Scikit-learn
* Flask
* Chart.js

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
