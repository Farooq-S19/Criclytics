Criclytics - AI-Powered Cricket Analytics Platform
Criclytics is a web-based cricket analytics tool built with Flask and powered by a machine learning model. It provides in-depth analysis for individual players and predicts match outcomes between two teams based on historical player data.
Live Demo / Screenshot
(It's highly recommended to add a screenshot or a GIF of the application in action here. It dramatically increases interest.)
<!-- Replace with your actual screenshot URL -->
Core Features
1. In-depth Player Analysis
Player Search: Select any player from the dataset to view their detailed profile.
AI-Powered Score Prediction: Utilizes a RandomForestRegressor model to predict a player's likely score range for an upcoming match, complete with a confidence score.
Dynamic Data Visualization:
Recent Performance: A bar chart showing a player's recent (simulated) scores.
Performance Breakdown: A pie chart illustrating how a player scores their runs (from 1s/2s/3s, 4s, and 6s).
Skills Radar: A radar chart that provides a visual summary of a player's key attributes like Consistency, Strike Rate, Power Hitting, and more.
Player Summary: A dynamically generated text summary of the player's style and key stats.
2. Team vs. Team Prediction
Head-to-Head Comparison: Select two teams to see a prediction of the match winner.
Win Probability: Displays the win probability for each team, calculated using a heuristic model based on aggregate player performance.
Key Players: Lists the top players for each team.
Team Statistics: Provides a summary of average team performance metrics (Avg. Runs, Avg. Strike Rate, etc.).
Prediction Reasoning: Offers a text-based explanation for why the AI predicted a certain outcome.
Technology Stack
Backend: Python, Flask
Machine Learning: Scikit-learn, Pandas, Joblib
Frontend: HTML, CSS, JavaScript (with Chart.js for data visualization)
Data: CSV file (players_data.csv) for static data storage.
Project Structure

Setup and Installation
Follow these steps to get the Criclytics application running on your local machine.
Prerequisites
Python 3.7+
pip (Python package installer)
1. Clone the Repository
code
Bash
git clone https://github.com/your-username/criclytics-flask.git
cd criclytics-flask
2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
On macOS/Linux:
code
Bash
python3 -m venv venv
source venv/bin/activate
On Windows:
code
Bash
python -m venv venv
.\venv\Scripts\activate
3. Install Dependencies
Install all the required Python libraries from the requirements.txt file.
code
Bash
pip install -r requirements.txt
4. Prepare the Data and Model (Crucial Step!)
The application relies on a dataset and a pre-trained machine learning model.
a. Add the Dataset:
Make sure you have your cricket player data in a file named players_data.csv.
Place this file inside the static/data/ directory.
b. Train the AI Model:
Before you can run the web app, you must first train the model using your data. Run the model.py script from your terminal:
code
Bash
python model.py
This script will read players_data.csv, train the RandomForestRegressor model, and save two files: player_score_model.pkl and feature_scaler.pkl inside the models/ directory.
5. Run the Flask Application
Once the model files are generated, you can start the Flask server.
code
Bash
flask run

