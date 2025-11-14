# --- START OF FILE app.py ---

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import traceback # For detailed error logging
import os # To check for file existence
import numpy as np # Import numpy to potentially check types if needed

app = Flask(__name__)

# --- Global Data and Model Loading ---
# Initialize as empty to prevent NameError
player_data_df = pd.DataFrame() 
player_model = None
feature_scaler = None

try:
    data_file_path = os.path.join('static', 'data', 'players_data.csv')
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"players_data.csv not found at: {data_file_path}")

    player_data_df = pd.read_csv(data_file_path)
    
    # Define numeric columns that should be treated as numbers
    numeric_cols = ['Matches', 'Innings', 'Runs', 'Highest', 'Average', 'BF', 'SR', '100', '4', '6']
    
    # Convert numeric columns to numeric types, coercing errors to NaN
    for col in numeric_cols:
        player_data_df[col] = pd.to_numeric(player_data_df[col], errors='coerce')
    
    # Fill any NaNs in these numeric columns with 0. This is crucial for calculations.
    player_data_df[numeric_cols] = player_data_df[numeric_cols].fillna(0)

    print("players_data.csv loaded and cleaned successfully.")
    # print(f"DataFrame Head:\n{player_data_df.head()}") # Uncomment to see loaded data
    # print(f"DataFrame Info:\n{player_data_df.info()}") # Uncomment to check data types

except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    print("Please ensure 'players_data.csv' is in 'criclytics-flask/static/data/'.")
    exit() # Exit if critical data file is missing
except Exception as e:
    print(f"FATAL ERROR: Could not load or process players_data.csv: {e}")
    traceback.print_exc()
    exit() # Exit if data loading fails critically


try:
    player_model_path = os.path.join('models', 'player_score_model.pkl')
    scaler_path = os.path.join('models', 'feature_scaler.pkl')

    if not os.path.exists(player_model_path) or not os.path.exists(scaler_path):
        print("WARNING: Model .pkl files not found. Player prediction will not work.")
        print("Please run 'python model.py' script first to train and save the models.")
    else:
        player_model = joblib.load(player_model_path)
        feature_scaler = joblib.load(scaler_path)
        print("AI models loaded successfully.")
except Exception as e:
    print(f"WARNING: Error loading AI models: {e}. Player prediction will not work.")
    traceback.print_exc()


# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/player-analysis')
def player_analysis_page():
    # Placeholder for model accuracy - replace with your actual model's validated accuracy
    model_accuracy = 85.7
    return render_template('player_analysis.html', model_accuracy=model_accuracy)

@app.route('/team-prediction')
def team_prediction_page():
    return render_template('team_prediction.html')

@app.route('/get_initial_data')
def get_initial_data():
    if player_data_df.empty:
        return jsonify({'players': [], 'teams': []})

    players = sorted(player_data_df['Name'].unique().tolist())
    teams = sorted(player_data_df['Team'].unique().tolist())
    return jsonify({'players': players, 'teams': teams})

@app.route('/get_player_data', methods=['POST'])
def get_player_data():
    player_name = request.json.get('player_name')
    print(f"Request received for player: {player_name}")

    if player_data_df.empty:
        print("Error: player_data_df is empty. Cannot fetch player data.")
        return jsonify({"error": "No player data available."}), 500

    try:
        player_row_series = player_data_df[player_data_df['Name'] == player_name]
        if player_row_series.empty:
            print(f"Player '{player_name}' not found in data.")
            return jsonify({"error": "Player not found."}), 404
        
        # Accessing single row as a Series; values from it can be numpy types (e.g., np.int64)
        player_row = player_row_series.iloc[0] 
        # print(f"Player row found:\n{player_row}") # Uncomment for full row debugging

        # --- Player Profile Data (Using .get() for robustness, no explicit cast needed here for string values) ---
        player_profile = {
            "name": str(player_row.get('Name', 'Unknown Player')), # Cast to str just in case
            "team": str(player_row.get('Team', 'N/A')),
            "role": str(player_row.get('Batting_Hand', 'N/A')),
            "nationality": "International", # Placeholder if not in CSV, or fetch from CSV if available
            "age": int(player_row.get('Age', 28)) # Cast age to standard int
        }
        
        # Use .get() with default for safety in string formatting
        player_summary_avg = float(player_row.get('Average', 0)) # Cast to float
        player_summary_sr = float(player_row.get('SR', 0)) # Cast to float

        player_summary = f"{player_profile['name']} is a {player_profile['role']} batsman. With a solid average of {player_summary_avg:.2f}, they are a reliable pillar for the team."
        if player_summary_sr > 150: # Adjust threshold as needed
            player_summary = f"{player_profile['name']} is a {player_profile['role']} batsman. With an explosive strike rate of {player_summary_sr:.2f}, they are a significant threat."

        # --- AI Prediction ---
        player_prediction = {
            "score_range": "N/A",
            "confidence": "0%"
        }
        if player_model and feature_scaler:
            try:
                # CRITICAL FIX: Ensure these features and their ORDER EXACTLY match what your model was trained on
                # Based on your model.py: ['Average', 'SR', 'BF', 'Highest', '100', '6']
                features_for_model = pd.DataFrame([[
                    float(player_row.get('Average', 0)), # Cast to float
                    float(player_row.get('SR', 0)),      # Cast to float
                    float(player_row.get('BF', 0)),      # Cast to float
                    float(player_row.get('Highest', 0)), # Cast to float
                    int(player_row.get('100', 0)),       # Cast to int
                    int(player_row.get('6', 0))          # Cast to int
                ]], columns=['Average', 'SR', 'BF', 'Highest', '100', '6'])
                
                # The .astype(float) here is still good for consistency
                features_for_model = features_for_model.astype(float) 

                scaled_features = feature_scaler.transform(features_for_model)
                predicted_runs = player_model.predict(scaled_features)[0]
                
                # Generate a range for prediction (e.g., +/- 15% of prediction for variability)
                lower_bound = max(0, int(predicted_runs * 0.85)) # Final result is standard int
                upper_bound = int(predicted_runs * 1.15)         # Final result is standard int
                
                # Dummy confidence - replace with actual model confidence if available
                confidence_val = f"{min(95, max(50, int(predicted_runs * 1.5) % 100))}%"
                if predicted_runs < 10: confidence_val = "60%" 
                
                player_prediction = {
                    "score_range": f"{lower_bound} - {upper_bound} runs",
                    "confidence": confidence_val
                }
                print(f"AI Prediction: {player_prediction}")

            except Exception as e:
                print(f"Error during AI prediction for {player_name}: {e}")
                traceback.print_exc()
        else:
            print("AI models are not loaded or failed to load. Skipping prediction for player.")


        # --- Chart Data ---
        # Recent Performance (Bar Chart) - Dummy data for demonstration
        player_avg = float(player_row.get('Average', 0)) # Cast to float

        recent_performance_data = {
            "labels": ["M1", "M2", "M3", "M4", "M5"],
            "data": [
                int(max(0, round(player_avg * 0.8, 1))), # Cast to float
                int(max(0, round(player_avg * 0.3, 1))), # Cast to float
                int(max(0, round(player_avg * 1.2, 1))), # Cast to float
                int(max(0, round(player_avg * 0.5, 1))), # Cast to float
                int(max(0, round(player_avg * 0.9, 1)))  # Cast to float
            ]
        }

        # Performance Breakdown (Pie Chart) - Based on Runs, 4s, 6s (ENHANCED)
        total_runs_actual = int(player_row.get('Runs', 0)) # Cast to int
        runs_from_4s = int(player_row.get('4', 0)) * 4      # Cast to int
        runs_from_6s = int(player_row.get('6', 0)) * 6      # Cast to int
        
        runs_from_other = total_runs_actual - (runs_from_4s + runs_from_6s)
        runs_from_other = max(0, int(runs_from_other)) # Ensure non-negative and cast to int

        if total_runs_actual == 0:
            runs_from_other = 0
            runs_from_4s = 0
            runs_from_6s = 0

        performance_breakdown_data = {
            "labels": ["Runs (1s/2s/3s)", "Runs from 4s", "Runs from 6s"],
            "data": [runs_from_other, runs_from_4s, runs_from_6s]
        }

        # Skills Radar (Radar Chart) - Dummy data for demonstration (0-100 scale)
        # Using .get() and max(1, ...) for denominators to prevent errors
        # Explicitly casting to int/float for safety
        player_avg_safe = float(player_row.get('Average', 0))
        player_sr_safe = float(player_row.get('SR', 0))
        player_bf_safe = float(player_row.get('BF', 0))
        player_6s_safe = int(player_row.get('6', 0))
        player_runs_safe = float(player_row.get('Runs', 0))
        player_innings_safe = float(player_row.get('Innings', 0))
        player_matches_safe = float(player_row.get('Matches', 0))

        consistency_score = int(min(100, max(20, player_avg_safe * 2.5)))
        strike_rate_score = int(min(100, max(20, player_sr_safe * 0.7)))
        
        power_hitting_score = 0
        if player_bf_safe > 0:
            power_hitting_score = int(min(100, max(20, (player_6s_safe * 100 / player_bf_safe) * 5)))
        
        versatility_score = int(min(100, max(20, (player_runs_safe / max(1, player_innings_safe)) * 1.5)))
        finishing_score = int(min(100, max(20, (player_runs_safe / max(1, player_matches_safe)) * 1.8)))

        skills_radar_data = {
            "labels": ["Consistency", "Strike Rate", "Power Hitting", "Versatility", "Finishing"],
            "data": [
                consistency_score,
                strike_rate_score,
                power_hitting_score,
                versatility_score,
                finishing_score
            ]
        }
        # Clamp radar values to 0-100 after calculations and ensure they are standard ints
        skills_radar_data['data'] = [int(max(0, min(100, val))) for val in skills_radar_data['data']]


        return jsonify({
            "profile": player_profile,
            "summary": player_summary,
            "prediction": player_prediction,
            "charts": {
                "recent_performance": recent_performance_data,
                "performance_breakdown": performance_breakdown_data,
                "skills_radar": skills_radar_data
            }
        })

    except Exception as e:
        print(f"--- AN UNEXPECTED ERROR OCCURRED IN /get_player_data ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc() # Prints the full traceback for more context
        print(f"-------------------------------------------------------")
        # Return a generic error to frontend for the alert message
        return jsonify({"error": "Internal Server Error", "details": "An unexpected error occurred on the server."}), 500

@app.route('/get_team_prediction', methods=['POST'])
def get_team_prediction():
    team1_name = request.json.get('team1')
    team2_name = request.json.get('team2')
    print(f"Request received for team prediction: {team1_name} vs {team2_name}")

    if player_data_df.empty:
        print("Error: player_data_df is empty. Cannot perform team prediction.")
        return jsonify({"error": "No data available for team prediction."}), 500

    try:
        winner_team = ""
        winning_prob = 0
        prob_team1 = 0
        prob_team2 = 0
        prediction_reasoning = ""

        # Basic dummy logic: CSK vs DC, MI vs RCB fixed, others use average runs
        if team1_name == "Chennai Super Kings" and team2_name == "Delhi Capitals":
            winner_team = "Chennai Super Kings"
            winning_prob = 53
            prob_team1 = 47
            prob_team2 = 53
            prediction_reasoning = "Chennai Super Kings demonstrates greater consistency and higher average performance metrics across their key players."
        elif team1_name == "Royal Challengers Bangalore" and team2_name == "Mumbai Indians":
            winner_team = "Mumbai Indians"
            winning_prob = 58
            prob_team1 = 42
            prob_team2 = 58
            prediction_reasoning = "Mumbai Indians have a stronger head-to-head record and a more balanced squad depth."
        else:
            team1_avg_runs = player_data_df[player_data_df['Team'] == team1_name]['Runs'].mean()
            team2_avg_runs = player_data_df[player_data_df['Team'] == team2_name]['Runs'].mean()

            team1_avg_runs = float(team1_avg_runs) if pd.notna(team1_avg_runs) else 0 # Cast to float
            team2_avg_runs = float(team2_avg_runs) if pd.notna(team2_avg_runs) else 0 # Cast to float

            if team1_avg_runs > team2_avg_runs:
                winner_team = team1_name
                winning_prob = min(90, max(51, 50 + int((team1_avg_runs - team2_avg_runs) * 0.5)))
                prob_team1 = winning_prob
                prob_team2 = 100 - winning_prob
                prediction_reasoning = f"{team1_name} has a slight edge based on a higher average team performance from their players."
            elif team2_avg_runs > team1_avg_runs:
                winner_team = team2_name
                winning_prob = min(90, max(51, 50 + int((team2_avg_runs - team1_avg_runs) * 0.5)))
                prob_team1 = 100 - winning_prob
                prob_team2 = winning_prob
                prediction_reasoning = f"{team2_name} has a slight edge based on a higher average team performance from their players."
            else: # A tie or no significant difference
                winner_team = team1_name # Default to team1 for display
                winning_prob = 50
                prob_team1 = 50
                prob_team2 = 50
                prediction_reasoning = "The teams appear evenly matched based on current data."


        # --- Player Lists ---
        team1_players = player_data_df[player_data_df['Team'] == team1_name]['Name'].unique().tolist()
        team2_players = player_data_df[player_data_df['Team'] == team2_name]['Name'].unique().tolist()
        team1_players = sorted(team1_players)[:7]
        team2_players = sorted(team2_players)[:7]


        # --- Team Summary Data ---
        team1_relevant_players = player_data_df[player_data_df['Team'] == team1_name]
        team2_relevant_players = player_data_df[player_data_df['Team'] == team2_name]

        # Calculate means, handling potential NaNs and casting to standard Python types
        team1_avg_runs_player = float(team1_relevant_players['Runs'].mean()) if not team1_relevant_players.empty else 0
        team1_avg_sr_player = float(team1_relevant_players['SR'].mean()) if not team1_relevant_players.empty else 0
        team1_avg_4s_player = float(team1_relevant_players['4'].mean()) if not team1_relevant_players.empty else 0
        team1_avg_6s_player = float(team1_relevant_players['6'].mean()) if not team1_relevant_players.empty else 0

        team2_avg_runs_player = float(team2_relevant_players['Runs'].mean()) if not team2_relevant_players.empty else 0
        team2_avg_sr_player = float(team2_relevant_players['SR'].mean()) if not team2_relevant_players.empty else 0
        team2_avg_4s_player = float(team2_relevant_players['4'].mean()) if not team2_relevant_players.empty else 0
        team2_avg_6s_player = float(team2_relevant_players['6'].mean()) if not team2_relevant_players.empty else 0


        team1_summary = {
            "Avg Runs/Player": f"{team1_avg_runs_player:.0f}",
            "Avg SR/Player": f"{team1_avg_sr_player:.1f}",
            "Avg 4s/Player": f"{team1_avg_4s_player:.0f}",
            "Avg 6s/Player": f"{team1_avg_6s_player:.0f}"
        }
        team2_summary = {
            "Avg Runs/Player": f"{team2_avg_runs_player:.0f}",
            "Avg SR/Player": f"{team2_avg_sr_player:.1f}",
            "Avg 4s/Player": f"{team2_avg_4s_player:.0f}",
            "Avg 6s/Player": f"{team2_avg_6s_player:.0f}"
        }

        return jsonify({
            "prediction": {
                "winner": winner_team,
                "confidence": f"{winning_prob}%",
                "prob1": prob_team1,
                "prob2": prob_team2,
                "reason": prediction_reasoning
            },
            "team1_players": team1_players,
            "team2_players": team2_players,
            "team1_summary": team1_summary,
            "team2_summary": team2_summary
        })

    except Exception as e:
        print(f"--- AN UNEXPECTED ERROR OCCURRED IN /get_team_prediction ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()
        print(f"-------------------------------------------------------")
        return jsonify({"error": "Internal Server Error", "details": "An unexpected error occurred on the server."}), 500


if __name__ == '__main__':
    # BEFORE RUNNING:
    # 1. Ensure 'players_data.csv' is in static/data/
    # 2. Ensure your 'model.py' script has been run at least once
    #    to create 'models/player_score_model.pkl' and 'models/feature_scaler.pkl'
    #    with features: ['Average', 'SR', 'BF', 'Highest', '100', '6']
    
    app.run(debug=True) # debug=True is good for development, but set to False in production!

# END OF FILE app.py 
