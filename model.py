# --- START OF FILE model.py ---

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
import numpy as np

# --- Configuration ---
DATA_PATH = os.path.join('static', 'data', 'players_data.csv')
MODEL_DIR = 'models'
PLAYER_MODEL_PATH = os.path.join(MODEL_DIR, 'player_score_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.pkl')

def _load_and_clean_data():
    """A robust function to load and clean the specific CSV data."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, header=0)
    print(f"1. Successfully loaded CSV. Shape: {df.shape}")

    df.dropna(subset=['Name', 'Team'], inplace=True)
    df = df[df['Name'].str.strip() != '']
    print(f"2. After dropping rows with no Name/Team. Shape: {df.shape}")

    numeric_cols = ['Matches', 'Innings', 'Runs', 'Highest', 'Average', 'BF', 'SR', '100', '4', '6']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    print(f"3. After cleaning numeric columns. Shape: {df.shape}")
    return df

def train_player_model():
    """
    Trains a model to predict player performance and saves it.
    """
    print("\n--- Training Player Prediction Model ---")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    df = _load_and_clean_data()
    
    #  Filter for players with at least some batting innings
    train_df = df[df['Innings'] > 0].copy()
    print(f"4. Filtered for training (Innings > 0). Shape: {train_df.shape}")

    if train_df.empty:
        print("ERROR: No data left to train the model after filtering. Please check your 'Innings' column in players_data.csv.")
        return 0

    # Feature Engineering (if any, ensure these are also aligned)
    # The 'target' here is 'Runs', but the model will predict based on features
    # 'Predicted_Score' is an internal target for this specific training script
    train_df.loc[:, 'Calculated_Target_Score'] = (train_df['Average'] * 0.6) + (train_df['SR'] * 0.2) + (train_df['Highest'] * 0.1)
    
    # THESE ARE THE FEATURES YOUR MODEL WILL BE TRAINED ON AND SAVED WITH
    # Ensure this list EXACTLY matches what you pass to app.py for prediction
    features = ['Average', 'SR', 'BF', 'Highest', '100', '6'] # Corrected features as per discussion
    target = 'Calculated_Target_Score' # This is the target your model learns to predict

    X = train_df[features]
    y = train_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    # Define "accuracy" for regression as predictions within a certain error margin
    accuracy = np.mean(np.abs(predictions - y_test) < 15) * 100 
    print(f"5. Model 'Accuracy' (prediction within 15 runs): {accuracy:.2f}%")

    joblib.dump(model, PLAYER_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"6. Model and scaler saved successfully to '{MODEL_DIR}/'")
    
    return accuracy

# (The rest of the functions in model.py can remain the same as the last version,
#  but I'll include them for completeness with the updated feature set in get_ai_prediction for clarity)

def get_ai_prediction(player_data):
    """
    Uses the trained model to predict a player's score.
    """
    # This block usually isn't called directly from model.py's __main__ but from app.py
    # so the model loading should ideally be done once in app.py.
    # However, for robustness in model.py if called directly, it's fine.
    try:
        model = joblib.load(PLAYER_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Model files not found for prediction. Please run train_player_model() first.")
        return {"score_range": "N/A", "confidence": "0%"}
    
    # CRITICAL: These features MUST match the training features and their order
    features = ['Average', 'SR', 'BF', 'Highest', '100', '6'] 
    
    # Create DataFrame from player_data (dictionary)
    player_df = pd.DataFrame([player_data])
    
    # Ensure all feature columns exist and are numeric, fill NaNs
    for col in features:
        player_df[col] = pd.to_numeric(player_df.get(col, 0), errors='coerce').fillna(0)

    player_features = player_df[features]
    player_features_scaled = scaler.transform(player_features)
    
    predicted_score = model.predict(player_features_scaled)[0]
    
    lower_bound = max(0, int(predicted_score - 10))
    upper_bound = int(predicted_score + 15)
    
    # Dummy confidence calculation (adjust as per your actual model's confidence logic)
    confidence = min(95, 60 + (player_data.get('Average', 0) / 5) + (player_data.get('SR', 100) - 100) / 10)

    return {
        "score_range": f"{lower_bound} – {upper_bound} runs",
        "confidence": f"{min(95, max(65, int(confidence)))}%"
    }

def get_team_win_prediction(team1_name, team2_name, all_data):
    """
    Heuristic-based AI to predict team winner. (This is distinct from the player model)
    """
    def calculate_impact_score(player):
        if not isinstance(player, dict): return 0
        if player.get('Innings', 0) < 1: return 0 
        
        # Ensure all values used are numeric and handle potential NaNs
        avg = player.get('Average', 0)
        sr = player.get('SR', 0)
        runs = player.get('Runs', 0)
        innings = player.get('Innings', 0)
        sixes = player.get('6', 0)
        fours = player.get('4', 0)

        # Prevent ZeroDivisionError
        runs_per_inning = (runs / innings) if innings > 0 else 0
        
        score = (avg * 1.5) + (sr * 0.8) + \
                (runs_per_inning * 1.2) + \
                ((sixes * 2 + fours) * 0.5)
        return score

    team1_players = all_data[all_data['Team'] == team1_name].to_dict('records')
    team2_players = all_data[all_data['Team'] == team2_name].to_dict('records')

    team1_strength = sum(calculate_impact_score(p) for p in team1_players)
    team2_strength = sum(calculate_impact_score(p) for p in team2_players)

    total_strength = team1_strength + team2_strength
    if total_strength == 0:
        return {"winner": "Toss-up", "confidence": "50%", "reason": "Not enough data for a reliable prediction.", "prob1": 50, "prob2": 50}

    prob1 = (team1_strength / total_strength) * 100
    prob2 = 100 - prob1
    
    if prob1 > prob2:
        winner, confidence = team1_name, prob1
        reason = f"{team1_name} has a stronger overall player impact score, suggesting superior batting depth and power-hitting capability."
    else:
        winner, confidence = team2_name, prob2
        reason = f"{team2_name} demonstrates greater consistency and higher average performance metrics across their key players."
        
    return {"winner": winner, "confidence": f"{confidence:.0f}%", "reason": reason, "prob1": round(prob1), "prob2": round(prob2)}

if __name__ == '__main__':
    train_player_model()

# --- END OF FILE model.py ---