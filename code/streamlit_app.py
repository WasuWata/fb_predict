import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import json
import requests
import tempfile
import xgboost as xgb

# ML Model
response = requests.get('https://raw.githubusercontent.com/WasuWata/fb_predict/main/code/model/model.json')
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write(response.text)
            tmp_path = tmp.name

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softprob',  # For multi-class
    num_class=3,
    random_state=42
)
model._estimator_type = 'classifier'
model.load_model(tmp_path)

# Premier league data
url = 'https://raw.githubusercontent.com/WasuWata/fb_predict/main/code/source/summary.csv'
data = pd.read_csv(url)
teams = data['Team_Team'].unique().tolist()
player_dict = {}
for team in teams:
    team_players = data[data['Team_Team'] == team]['Unnamed: 0_level_0_Player'].unique().tolist()
    player_dict[team] = team_players

premier_league_data = {}
premier_league_data['teams'] = teams
premier_league_data['players'] = player_dict
premier_league_data['venues'] = ['Home','Away','Neutral']

# Common football formations with positions
formations = {
    "4-4-2": {
        "defenders": 4,
        "midfielders": 4,
        "forwards": 2,
        "positions": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    },
    "4-3-3": {
        "defenders": 4,
        "midfielders": 3,
        "forwards": 3,
        "positions": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "ST", "RW"]
    },
    "3-4-3": {
        "defenders": 3,
        "midfielders": 4,
        "forwards": 3,
        "positions": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "RWB", "LW", "ST", "RW"]
    },
    "4-2-3-1": {
        "defenders": 4,
        "midfielders": 5,
        "forwards": 1,
        "positions": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "CAM", "LW", "RW", "ST"]
    },
    "3-5-2": {
        "defenders": 3,
        "midfielders": 5,
        "forwards": 2,
        "positions": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "CM", "RWB", "ST", "ST"]
    },
    "5-3-2": {
        "defenders": 5,
        "midfielders": 3,
        "forwards": 2,
        "positions": ["GK", "LWB", "CB", "CB", "CB", "RWB", "CM", "CM", "CM", "ST", "ST"]
    }
}

# Position categories for ML features
position_categories = {
    "GK": "goalkeeper",
    "LB": "defender", "RB": "defender", "CB": "defender", "LWB": "defender", "RWB": "defender",
    "LM": "midfielder", "RM": "midfielder", "CM": "midfielder", "CDM": "midfielder", "CAM": "midfielder",
    "LW": "forward", "RW": "forward", "ST": "forward"
}

# Set page configuration
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for better styling with centered alignment
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #38003c;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #00ff87;
        font-size: 1.5rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ff87;
        margin: 10px 0;
    }
    .team-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .position-box-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 5px;
    }
    .position-box {
        background-color: white;
        border: 2px solid #00ff87;
        border-radius: 8px;
        padding: 10px;
        min-height: 60px;
        min-width: 80px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 5px;
    }
    .position-label {
        font-weight: bold;
        color: #38003c;
        font-size: 0.9rem;
    }
    .data-preview {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
        font-family: monospace;
        font-size: 0.85rem;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

def get_lineup_data(key_prefix, formation_name):
    """Extract and structure lineup data for ML model input"""
    if f'{key_prefix}_lineup' not in st.session_state:
        return None
    
    lineup_dict = st.session_state.get(f'{key_prefix}_lineup', {})
    formation_positions = formations[formation_name]["positions"]
    
    # Create structured lineup data
    lineup_data = {
        "formation": formation_name,
        "players": [],
        "positions": [],
        "position_categories": [],
        "count_by_position": {},
        "count_by_category": {}
    }
    
    # Extract player-position pairs
    for unique_key, player_name in lineup_dict.items():
        if player_name:  # Only include selected players
            # Extract base position (remove suffix like "_1", "_2")
            base_position = unique_key.split('_')[0]
            lineup_data["players"].append(player_name)
            lineup_data["positions"].append(base_position)
            lineup_data["position_categories"].append(position_categories.get(base_position, "unknown"))
    
    return lineup_data

def create_ml_features(home_lineup_data, away_lineup_data, home_team, away_team, venue, home_form, away_form):
    """Create feature vector for ML model prediction"""
    
    # Form to numerical mapping
    form_mapping = {"Terrible": 1, "Poor": 2, "Average": 3, "Good": 4, "Excellent": 5}
    
    # Venue to numerical mapping
    venue_mapping = {"Home": 1, "Away": 0, "Neutral": 0.5}
    
    # Base features
    features = {
        # Team identifiers
        "home_team": home_team,
        "away_team": away_team,
        
        # Match context
        "venue": venue_mapping.get(venue, 0.5),
        "home_form": form_mapping.get(home_form, 3),
        "away_form": form_mapping.get(away_form, 3),
        
        # Formation information
        "home_formation": home_lineup_data["formation"] if home_lineup_data else "unknown",
        "away_formation": away_lineup_data["formation"] if away_lineup_data else "unknown",
    }
    
    # Add lineup-based features if available
    if home_lineup_data:
        features.update({
            # Position counts for home team
            "home_gk_count": home_lineup_data["count_by_category"].get("goalkeeper", 0),
            "home_def_count": home_lineup_data["count_by_category"].get("defender", 0),
            "home_mid_count": home_lineup_data["count_by_category"].get("midfielder", 0),
            "home_fwd_count": home_lineup_data["count_by_category"].get("forward", 0),
            
            # Specific position counts for home team
            "home_cb_count": home_lineup_data["count_by_position"].get("CB", 0),
            "home_fullback_count": home_lineup_data["count_by_position"].get("LB", 0) + 
                                  home_lineup_data["count_by_position"].get("RB", 0) +
                                  home_lineup_data["count_by_position"].get("LWB", 0) +
                                  home_lineup_data["count_by_position"].get("RWB", 0),
            "home_striker_count": home_lineup_data["count_by_position"].get("ST", 0),
            
            # Player count
            "home_player_count": len(home_lineup_data["players"]),
        })
    
    if away_lineup_data:
        features.update({
            # Position counts for away team
            "away_gk_count": away_lineup_data["count_by_category"].get("goalkeeper", 0),
            "away_def_count": away_lineup_data["count_by_category"].get("defender", 0),
            "away_mid_count": away_lineup_data["count_by_category"].get("midfielder", 0),
            "away_fwd_count": away_lineup_data["count_by_category"].get("forward", 0),
            
            # Specific position counts for away team
            "away_cb_count": away_lineup_data["count_by_position"].get("CB", 0),
            "away_fullback_count": away_lineup_data["count_by_position"].get("LB", 0) + 
                                  away_lineup_data["count_by_position"].get("RB", 0) +
                                  away_lineup_data["count_by_position"].get("LWB", 0) +
                                  away_lineup_data["count_by_position"].get("RWB", 0),
            "away_striker_count": away_lineup_data["count_by_position"].get("ST", 0),
            
            # Player count
            "away_player_count": len(away_lineup_data["players"]),
        })
    
    # Calculate differences for comparative features
    if home_lineup_data and away_lineup_data:
        features.update({
            "def_count_diff": features.get("home_def_count", 0) - features.get("away_def_count", 0),
            "mid_count_diff": features.get("home_mid_count", 0) - features.get("away_mid_count", 0),
            "fwd_count_diff": features.get("home_fwd_count", 0) - features.get("away_fwd_count", 0),
        })
    
    return features

def create_formation_layout(team_name, formation_name, team_players, key_prefix):
    """Create a visual formation layout for team selection"""
    
    formation = formations[formation_name]
    positions = formation["positions"]
    
    # Initialize session state for player positions if not exists
    if f'{key_prefix}_lineup' not in st.session_state:
        st.session_state[f'{key_prefix}_lineup'] = {}
        # Initialize with unique keys for each position
        for i, pos in enumerate(positions):
            st.session_state[f'{key_prefix}_lineup'][f"{pos}_{i}"] = ""
    
    st.markdown(f"### {team_name} Formation: {formation_name}")
    
    # Create a copy of positions with unique indices for duplicate positions
    unique_positions = []
    position_count = {}
    for i, pos in enumerate(positions):
        if pos in position_count:
            position_count[pos] += 1
            unique_positions.append(f"{pos}_{position_count[pos]}")
        else:
            position_count[pos] = 1
            unique_positions.append(f"{pos}_1")
    
    # Pitch background
    with st.container():
        st.markdown('<div class="pitch-bg">', unsafe_allow_html=True)
        
        # Goalkeeper row
        st.markdown('<div class="formation-row gk-row">', unsafe_allow_html=True)
        # Find the goalkeeper position
        for i, pos in enumerate(positions):
            if pos == "GK":
                unique_key = unique_positions[i]
                create_position_box("GK", team_name, team_players, key_prefix, unique_key, i)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Group positions by row based on formation
        position_groups = group_positions_by_row(formation_name, positions, unique_positions)
        
        # Create each row
        for row_idx, row_positions in enumerate(position_groups):
            if row_positions:  # Skip empty rows
                st.markdown(f'<div class="formation-row">', unsafe_allow_html=True)
                
                # Create equal columns for each position in the row
                cols = st.columns(len(row_positions))
                
                for col_idx, (pos, unique_key, orig_idx) in enumerate(row_positions):
                    with cols[col_idx]:
                        # Create a container for centered alignment
                        st.markdown('<div class="position-box-container">', unsafe_allow_html=True)
                        create_position_box(pos, team_name, team_players, key_prefix, unique_key, orig_idx)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display data preview for debugging/verification
    with st.expander(f"View {team_name} Lineup Data (for ML model)"):
        lineup_data = get_lineup_data(key_prefix, formation_name)
        if lineup_data and len(lineup_data["players"]) > 0:
            st.markdown("**Structured Lineup Data:**")
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            
            # Display player-position mapping
            st.write("Player-Position Mapping:")
            for i, (player, position) in enumerate(zip(lineup_data["players"], lineup_data["positions"])):
                st.write(f"  {i+1}. {position}: {player}")
            
            st.write("\nPosition Counts:")
            for position, count in lineup_data["count_by_position"].items():
                st.write(f"  {position}: {count}")
            
            st.write("\nCategory Counts:")
            for category, count in lineup_data["count_by_category"].items():
                st.write(f"  {category}: {count}")
            
            st.write(f"\nTotal Players Selected: {len(lineup_data['players'])}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show JSON format (for API/ML model)
            st.markdown("**JSON Format:**")
            st.code(json.dumps(lineup_data, indent=2), language='json')
        else:
            st.info("No players selected yet. Select players to see lineup data.")

def group_positions_by_row(formation_name, positions, unique_positions):
    """Group positions into rows based on formation"""
    formation = formations[formation_name]
    defenders = formation["defenders"]
    midfielders = formation["midfielders"]
    forwards = formation["forwards"]
    
    position_groups = []
    
    # Skip GK (position 0)
    idx = 1
    
    # Defenders row
    defender_group = []
    for i in range(defenders):
        defender_group.append((positions[idx], unique_positions[idx], idx))
        idx += 1
    position_groups.append(defender_group)
    
    # Midfielders row
    midfielder_group = []
    for i in range(midfielders):
        midfielder_group.append((positions[idx], unique_positions[idx], idx))
        idx += 1
    position_groups.append(midfielder_group)
    
    # Forwards row
    forward_group = []
    for i in range(forwards):
        forward_group.append((positions[idx], unique_positions[idx], idx))
        idx += 1
    position_groups.append(forward_group)
    
    return position_groups

def create_position_box(position, team_name, team_players, key_prefix, unique_key, index):
    """Create a single position box with player dropdown"""
    
    # Create a container for centered content
    container = st.container()
    
    with container:
        # Position box
        st.markdown(f"""
        <div class="position-box">
            <div class="position-label">{position}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a dropdown for this position
        available_players = [""] + team_players
        
        # Get current selection from session state
        current_selection = st.session_state[f'{key_prefix}_lineup'].get(unique_key, "")
        
        # Create dropdown with unique key
        selected_player = st.selectbox(
            f"Select player for {position}",
            available_players,
            index=available_players.index(current_selection) if current_selection in available_players else 0,
            key=f"{key_prefix}_{unique_key}",
            label_visibility="collapsed"
        )
        
        # Update session state
        if selected_player != st.session_state[f'{key_prefix}_lineup'].get(unique_key):
            st.session_state[f'{key_prefix}_lineup'][unique_key] = selected_player

def predict_match(home_team, away_team, home_lineup_data, away_lineup_data, venue, form, use_ml=False):
    """Prediction function that can use either simple heuristic or ML model"""
    
    if use_ml and home_lineup_data and away_lineup_data:
        # This is where you would call your ML model
        # For now, we'll use a more sophisticated heuristic based on lineup data
        return predict_with_lineup_data(home_team, away_team, home_lineup_data, away_lineup_data, venue, form)
    else:
        # Fall back to simple heuristic
        return simple_predict(home_team, away_team, home_lineup_data, away_lineup_data, venue, form) # WILL NOT USE IT

def simple_predict(home_team, away_team, home_lineup_data, away_lineup_data, venue, form): # WILL NOT USE IT
    """Simple prediction function"""
    np.random.seed(hash(home_team + away_team) % 10000)
    
    home_base = np.random.randint(60, 90)
    away_base = np.random.randint(60, 90)
    
    # Venue advantage
    if venue == "Home":
        home_base += 10
    elif venue == "Away":
        away_base += 5
    
    # Form factor
    form_bonus = {"Excellent": 10, "Good": 5, "Average": 0, "Poor": -5, "Terrible": -10}
    home_base += form_bonus.get(form['home'], 0)
    away_base += form_bonus.get(form['away'], 0)
    
    # Calculate probabilities
    total = home_base + away_base + 30
    home_win_prob = home_base / total * 100
    away_win_prob = away_base / total * 100
    draw_prob = 100 - home_win_prob - away_win_prob
    
    # Determine winner
    if home_win_prob > away_win_prob and home_win_prob > 35:
        winner = home_team
        confidence = home_win_prob
    elif away_win_prob > home_win_prob and away_win_prob > 35:
        winner = away_team
        confidence = away_win_prob
    else:
        winner = "Draw"
        confidence = draw_prob
    
    return {
        'home_win_prob': round(home_win_prob, 1),
        'away_win_prob': round(away_win_prob, 1),
        'draw_prob': round(draw_prob, 1),
        'predicted_winner': winner,
        'confidence': round(confidence, 1),
        'method': 'heuristic'
    }

def predict_with_lineup_data(home_team, away_team, home_lineup_data, away_lineup_data, venue, form): # Need to be changed
    """More sophisticated prediction using lineup data"""
    
    # Start with base scores
    home_score = 50
    away_score = 50
    
    # Venue advantage
    venue_bonus = {"Home": 15, "Away": 5, "Neutral": 0}
    home_score += venue_bonus.get(venue, 0)
    
    # Form factor
    form_value = {"Excellent": 20, "Good": 10, "Average": 0, "Poor": -10, "Terrible": -20}
    home_score += form_value.get(form['home'], 0)
    away_score += form_value.get(form['away'], 0)
    
    # Analyze formations (certain formations counter others)
    formation_matchup = {
        ("4-4-2", "4-3-3"): 5,  # 4-3-3 generally counters 4-4-2
        ("3-5-2", "4-4-2"): 10, # 3-5-2 overloads midfield against 4-4-2
        ("4-2-3-1", "4-3-3"): -5, # 4-3-3 presses high against 4-2-3-1
    }
    
    matchup_key = (home_lineup_data["formation"], away_lineup_data["formation"])
    if matchup_key in formation_matchup:
        home_score += formation_matchup[matchup_key]
    elif (matchup_key[1], matchup_key[0]) in formation_matchup:
        away_score += formation_matchup[(matchup_key[1], matchup_key[0])]
    
    # Defensive strength bonus
    home_def_count = home_lineup_data["count_by_category"].get("defender", 0)
    away_def_count = away_lineup_data["count_by_category"].get("defender", 0)
    
    if home_def_count >= 4:
        home_score += 5
    if away_def_count >= 4:
        away_score += 5
    
    # Attacking strength bonus
    home_fwd_count = home_lineup_data["count_by_category"].get("forward", 0)
    away_fwd_count = away_lineup_data["count_by_category"].get("forward", 0)
    
    if home_fwd_count >= 3:
        home_score += 8
    elif home_fwd_count == 2:
        home_score += 5
    
    if away_fwd_count >= 3:
        away_score += 8
    elif away_fwd_count == 2:
        away_score += 5
    
    # Midfield control bonus
    home_mid_count = home_lineup_data["count_by_category"].get("midfielder", 0)
    away_mid_count = away_lineup_data["count_by_category"].get("midfielder", 0)
    
    if home_mid_count > away_mid_count:
        home_score += (home_mid_count - away_mid_count) * 3
    
    # Calculate final probabilities
    total = home_score + away_score + 40  # Extra for draw possibility
    home_win_prob = home_score / total * 100
    away_win_prob = away_score / total * 100
    draw_prob = 100 - home_win_prob - away_win_prob
    
    # Determine winner
    if home_win_prob > away_win_prob and home_win_prob > 40:
        winner = home_team
        confidence = home_win_prob
    elif away_win_prob > home_win_prob and away_win_prob > 40:
        winner = away_team
        confidence = away_win_prob
    else:
        winner = "Draw"
        confidence = draw_prob
    
    return {
        'home_win_prob': round(home_win_prob, 1),
        'away_win_prob': round(away_win_prob, 1),
        'draw_prob': round(draw_prob, 1),
        'predicted_winner': winner,
        'confidence': round(confidence, 1),
        'method': 'lineup_analysis',
        'features': create_ml_features(home_lineup_data, away_lineup_data, home_team, away_team, venue, form['home'], form['away'])
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Premier League Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict match outcomes using team formations and player data")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        menu_option = st.radio(
            "Choose an option:",
            ["Match Prediction"]
        )
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("App for predicting the Premier League's match outcome, based on player performance. (Using the data from 2017-Now)")
    
    if menu_option == "Match Prediction":
        # Main content - Match prediction form
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            st.markdown("### 🏠 Home Team")
            home_team = st.selectbox(
                "Select Home Team",
                premier_league_data["teams"],
                key="home_team"
            )
            
            if home_team:
                # Formation selection for home team
                home_formation = st.selectbox(
                    "Select Formation",
                    list(formations.keys()),
                    key="home_formation"
                )
                
                home_players_list = premier_league_data["players"].get(home_team, ["No player data available"])
                
                # Create formation layout for home team
                create_formation_layout(home_team, home_formation, home_players_list, "home")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            st.markdown("### 🚗 Away Team")
            away_team = st.selectbox(
                "Select Away Team",
                [team for team in premier_league_data["teams"] if team != home_team],
                key="away_team"
            )
            
            if away_team:
                # Formation selection for away team
                away_formation = st.selectbox(
                    "Select Formation",
                    list(formations.keys()),
                    key="away_formation"
                )
                
                away_players_list = premier_league_data["players"].get(away_team, ["No player data available"])
                
                # Create formation layout for away team
                create_formation_layout(away_team, away_formation, away_players_list, "away")
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional match details
        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            venue = st.selectbox(
                "Match Venue",
                premier_league_data["venues"],
                help="Select where the match is being played"
            )
        
        with col4:
            match_date = st.date_input(
                "Match Date",
                value=date.today()
            )
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button(
            "🔮 Predict Match Outcome",
            type="primary",
            use_container_width=True
        )
        
        if predict_button:
            # Get lineup data for ML model
            home_lineup_data = get_lineup_data("home", home_formation)
            away_lineup_data = get_lineup_data("away", away_formation)
            
            # Get prediction
            form_data = {'home': home_form, 'away': away_form}
            prediction = predict_match(
                home_team, away_team, 
                home_lineup_data, away_lineup_data, 
                venue, form_data, 
                use_ml=use_ml_prediction
            )
            
            # Display prediction results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("## 📊 Prediction Results")
            
            # Display prediction method
            st.info(f"Prediction method: {prediction.get('method', 'heuristic').replace('_', ' ').title()}")
            
            # Display formations
            col_formation1, col_formation2 = st.columns(2)
            with col_formation1:
                st.markdown(f"**{home_team} Formation:** {home_formation}")
                if home_lineup_data:
                    for player, position in zip(home_lineup_data["players"], home_lineup_data["positions"]):
                        st.write(f"• {position}: {player}")
            
            with col_formation2:
                st.markdown(f"**{away_team} Formation:** {away_formation}")
                if away_lineup_data:
                    for player, position in zip(away_lineup_data["players"], away_lineup_data["positions"]):
                        st.write(f"• {position}: {player}")
            
            st.markdown("---")
            
            # Winner prediction
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    label=f"🏆 Predicted Winner",
                    value=prediction['predicted_winner'],
                    delta=f"{prediction['confidence']}% confidence"
                )
            
            with col_b:
                st.metric(
                    label=f"📈 {home_team} Win Probability",
                    value=f"{prediction['home_win_prob']}%"
                )
            
            with col_c:
                st.metric(
                    label=f"📈 {away_team} Win Probability",
                    value=f"{prediction['away_win_prob']}%"
                )
            
            # Probability bars
            st.markdown("### Win Probability Distribution")
            prob_data = pd.DataFrame({
                'Outcome': [home_team, 'Draw', away_team],
                'Probability': [
                    prediction['home_win_prob'], 
                    prediction['draw_prob'], 
                    prediction['away_win_prob']
                ]
            })
            st.bar_chart(prob_data.set_index('Outcome'))
            
            # Show ML features if requested
            if show_ml_features and 'features' in prediction:
                with st.expander("View ML Features Used"):
                    st.markdown('<div class="data-preview">', unsafe_allow_html=True)
                    for key, value in prediction['features'].items():
                        st.write(f"{key}: {value}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Download features as JSON
                    features_json = json.dumps(prediction['features'], indent=2)
                    st.download_button(
                        label="Download Features as JSON",
                        data=features_json,
                        file_name=f"match_features_{home_team}_vs_{away_team}.json",
                        mime="application/json"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()