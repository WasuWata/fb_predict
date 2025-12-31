import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import json
import requests
import tempfile
import xgboost as xgb
import joblib
from io import BytesIO

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
model.n_classes_ = 3  # Set this based on your actual number of classes
model.classes_ = np.array([0, 1, 2])  # Assuming classes are 0, 1, 2 (Home, Draw, Away)
# Scaler
scaler_url = 'https://raw.githubusercontent.com/WasuWata/fb_predict/main/code/scaler/scaler.pkl'
response = requests.get(scaler_url)
scaler = joblib.load(BytesIO(response.content))

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
# formations = {
#     "4-4-2": {
#         "defenders": 4,
#         "midfielders": 4,
#         "forwards": 2,
#         "positions": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
#     },
#     "4-3-3": {
#         "defenders": 4,
#         "midfielders": 3,
#         "forwards": 3,
#         "positions": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "ST", "RW"]
#     },
#     "3-4-3": {
#         "defenders": 3,
#         "midfielders": 4,
#         "forwards": 3,
#         "positions": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "RWB", "LW", "ST", "RW"]
#     },
#     "4-2-3-1": {
#         "defenders": 4,
#         "midfielders": 5,
#         "forwards": 1,
#         "positions": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "CAM", "LW", "RW", "ST"]
#     },
#     "3-5-2": {
#         "defenders": 3,
#         "midfielders": 5,
#         "forwards": 2,
#         "positions": ["GK", "CB", "CB", "CB", "LWB", "CM", "CM", "CM", "RWB", "ST", "ST"]
#     },
#     "5-3-2": {
#         "defenders": 5,
#         "midfielders": 3,
#         "forwards": 2,
#         "positions": ["GK", "LWB", "CB", "CB", "CB", "RWB", "CM", "CM", "CM", "ST", "ST"]
#     }
# }

formations = {
    "4-4-2": {
        "defenders": 4,
        "midfielders": 4,
        "forwards": 2,
        "positions": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "FW", "FW"]
    },
    "4-3-3": {
        "defenders": 4,
        "midfielders": 3,
        "forwards": 3,
        "positions": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "FW", "RW"]
    },
    "3-4-3": {
        "defenders": 3,
        "midfielders": 4,
        "forwards": 3,
        "positions": ["GK", "CB", "CB", "CB", "LB", "CM", "CM", "WB", "LW", "FW", "RW"]
    },
    "4-2-3-1": {
        "defenders": 4,
        "midfielders": 5,
        "forwards": 1,
        "positions": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "RW", "FW"]
    },
    "3-5-2": {
        "defenders": 3,
        "midfielders": 5,
        "forwards": 2,
        "positions": ["GK", "CB", "CB", "CB", "WB", "CM", "CM", "CM", "WB", "FW", "FW"]
    },
    "5-3-2": {
        "defenders": 5,
        "midfielders": 3,
        "forwards": 2,
        "positions": ["GK", "WB", "CB", "CB", "CB", "WB", "CM", "CM", "CM", "FW", "FW"]
    }
}
# Position categories for ML features
position_categories = {
    "GK": "goalkeeper",
    "LB": "defender", "RB": "defender", "CB": "defender", "LWB": "defender", "RWB": "defender",
    "LM": "midfielder", "RM": "midfielder", "CM": "midfielder", "CDM": "midfielder", "CAM": "midfielder",
    "LW": "forward", "RW": "forward", "FW": "forward"
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

def get_lineup_data(key_prefix): # Done (maybe)
    """Extract and structure lineup data for ML model input"""
    if f'{key_prefix}_lineup' not in st.session_state:
        return None
    
    lineup_dict = st.session_state.get(f'{key_prefix}_lineup', {})
    # formation_positions = formations[formation_name]["positions"]
    
    # Create structured lineup data
    lineup_data = {
        "Unnamed: 0_level_0_Player": [],
        "Unnamed: 3_level_0_Pos": []
    }
    
    # Extract player-position pairs
    for unique_key, player_name in lineup_dict.items():
        if player_name and player_name.strip():  # Only include selected players
            # Extract base position (remove suffix like "_1", "_2")
            base_position = unique_key.split('_')[0]
            lineup_data["Unnamed: 0_level_0_Player"].append(player_name)
            lineup_data["Unnamed: 3_level_0_Pos"].append(base_position)
    
    return lineup_data

def extract_team_features(df, is_home = True): # Done (maybe)
    team_df = []
    for i in range(len(df['Unnamed: 0_level_0_Player'])):
        # player_df = data[(data['Unnamed: 0_level_0_Player'] == df['Unnamed: 0_level_0_Player'][i]) & (data['Unnamed: 3_level_0_Pos'] == df['Unnamed: 3_level_0_Pos'][i])].iloc[-3:,:]
        player_df = data[data['Unnamed: 0_level_0_Player'] == df['Unnamed: 0_level_0_Player'][i]].iloc[-3:,:]
        # player_position = player_df['Unnamed: 3_level_0_Pos'].iloc[0] if not player_df.empty else None
        player_position = df['Unnamed: 3_level_0_Pos'][i]
        player_df_average = player_df.groupby('Unnamed: 0_level_0_Player').mean(numeric_only = True)
        player_df_average['Team_Team'] = player_df['Team_Team'].unique()[-1]
        player_df_average['Unnamed: 3_level_0_Pos'] = player_position
        team_df.append(player_df_average)
    team_df = pd.concat(team_df)

    team_df['Performance.4_Sh'] = team_df['Performance.4_Sh'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']
    team_df['Performance.5_SoT'] = team_df['Performance.5_SoT'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']
    team_df['SCA_SCA'] = team_df['SCA_SCA'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']
    team_df['Performance.9_Tkl'] = team_df['Performance.9_Tkl'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']
    team_df['Performance.10_Int'] = team_df['Performance.10_Int'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']
    team_df['Performance.11_Blocks'] = team_df['Performance.11_Blocks'].astype('float32')*90/team_df['Unnamed: 5_level_0_Min']

    features = {}
    features['avg_minutes'] = team_df['Unnamed: 5_level_0_Min'].astype('float32').mean()
    # Shooting
    features['total_shots'] = team_df['Performance.4_Sh'].astype('float32').sum()
    features['shots_on_target'] = team_df['Performance.5_SoT'].astype('float32').sum()
    features['xG'] = team_df['Expected_xG'].astype('float32').sum()
    features['xAG'] = team_df['Expected.2_xAG'].astype('float32').sum()
    # Passing
    features['key_passes'] = team_df['SCA_SCA'].astype('float32').sum() # Shot creating action
    features['pass_completion'] = team_df['Passes_Cmp'].astype('float32').sum()/team_df['Passes.1_Att'].astype('float32').sum()*100

    # Defensive
    features['tackles'] = team_df['Performance.9_Tkl'].astype('float32').sum()
    features['interception'] = team_df['Performance.10_Int'].astype('float32').sum()
    features['blocks'] = team_df['Performance.11_Blocks'].astype('float32').sum()

    # Cards
    features['yellow_cards'] = team_df['Performance.6_CrdY'].astype('float32').sum()
    features['red_cards'] = team_df['Performance.7_CrdR'].astype('float32').sum()

    # Position-specific
    positions = team_df['Unnamed: 3_level_0_Pos'].astype(str)

    # Attackers (FW, LW, RW, ST)
    attackers = team_df[positions.str.contains('FW|LW|RW|ST|AM')]

    if len(attackers) > 0:
        features['attackers_xG'] = attackers['Expected_xG'].astype('float32').sum()
        features['attackers_shots'] = attackers['Performance.4_Sh'].astype('float32').sum()

    midfielders = team_df[positions.str.contains('CM|DM|LM|RM|AM')]
    
    if len(midfielders) > 0:
        features['midfielders_passes'] = midfielders['Passes_Cmp'].astype('float32').sum()/midfielders['Passes.1_Att'].astype('float32').sum()*100

    defenders = team_df[positions.str.contains('CB|RB|LB|WB|DF')]

    if len(defenders) > 0:
        features['defenders_tackles'] = defenders['Performance.9_Tkl'].astype('float32').sum()
        features['defenders_blocks'] = defenders['Performance.11_Blocks'].astype('float32').sum()
    return features

def process_match_file(home_df, away_df): # Done (maybe)
    home_data = home_df
    away_data = away_df
    home_features = extract_team_features(home_data, is_home = True)
    away_features = extract_team_features(away_data, is_home = False)

    match_features = {}
    for key, value in home_features.items():
        match_features[f'home_{key}'] = value

    for key, value in away_features.items():
        match_features[f'away_{key}'] = value

    for key in home_features.keys():
        if key in away_features:
            match_features[f'diff_{key}'] = home_features[key] - away_features[key]
            match_features[f'ratio_{key}'] = home_features[key]/(away_features[key] + 0.00001) # avoid dividing by zero

    return match_features 

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
            "home_player_count": len(home_lineup_data["Unnamed: 0_level_0_Player"]),
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
            "away_player_count": len(away_lineup_data["Unnamed: 0_level_0_Player"]),
        })
    
    # Calculate differences for comparative features
    if home_lineup_data and away_lineup_data:
        features.update({
            "def_count_diff": features.get("home_def_count", 0) - features.get("away_def_count", 0),
            "mid_count_diff": features.get("home_mid_count", 0) - features.get("away_mid_count", 0),
            "fwd_count_diff": features.get("home_fwd_count", 0) - features.get("away_fwd_count", 0),
        })
    
    return features

def create_formation_layout(team_name, formation_name, team_players, key_prefix): # Done
    """Create a visual formation layout for team selection"""
    
    formation = formations[formation_name]
    positions = formation["positions"]
    
    # Initialize session state for player positions if not exists
    # if f'{key_prefix}_lineup' not in st.session_state:
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
        lineup_data = get_lineup_data(key_prefix)
        if lineup_data and len(lineup_data["Unnamed: 0_level_0_Player"]) > 0:
            st.markdown("**Structured Lineup Data:**")
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            
            # Display player-position mapping
            st.write("Player-Position Mapping:")
            for i, (player, position) in enumerate(zip(lineup_data["Unnamed: 0_level_0_Player"], lineup_data["Unnamed: 3_level_0_Pos"])):
                st.write(f"  {i+1}. {position}: {player}")

            
            st.write(f"\nTotal Players Selected: {len(lineup_data['Unnamed: 0_level_0_Player'])}")
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
    widget_key = f"{key_prefix}_{unique_key}"
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
        if f'{key_prefix}_lineup' not in st.session_state:
            st.session_state[f'{key_prefix}_lineup'] = {}
    
        # Update the dictionary with the widget's value
        st.session_state[f'{key_prefix}_lineup'][unique_key] = selected_player
        # # Update session state
        # if selected_player != st.session_state[f'{key_prefix}_lineup'].get(unique_key):
        #     st.session_state[f'{key_prefix}_lineup'][unique_key] = selected_player

def predict_match():
    """Prediction function that can use either simple heuristic or ML model"""
    return predict_with_lineup_data()

def predict_with_lineup_data(): # Need to be changed
    """More sophisticated prediction using lineup data"""
    home_df = get_lineup_data('home')
    away_df = get_lineup_data('away')
    X = process_match_file(home_df,away_df)
    X = pd.DataFrame([X])
    X_scaled = scaler.transform(X)
    match_result = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)
    return {
        'home_win_prob': round(prob[0][0], 1),
        'away_win_prob': round(prob[0][2], 1),
        'draw_prob': round(prob[0][1], 1),
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
            home_lineup_data = get_lineup_data("home")
            away_lineup_data = get_lineup_data("away")
            
            st.write("### 🐛 DEBUG: Session State Before Prediction")
            st.write(f"Keys in session_state: {list(st.session_state.keys())}")    
            st.write(f'home_lineup_data: {home_lineup_data}')
            st.write(f'away_lineup_data: {away_lineup_data}')
            # Get lineup data for ML model
            # team_df = extract_team_features(home_lineup_data)
            # st.write(f"DEBUG: All positions in team_df: {team_df['Unnamed: 3_level_0_Pos'].unique()}")
    
            X = process_match_file(home_lineup_data,away_lineup_data) 
            st.write(f'X: {X}')
            # Get prediction
            prediction = predict_match()
            
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
                    for player, position in zip(home_lineup_data["Unnamed: 0_level_0_Player"], home_lineup_data["Unnamed: 3_level_0_Pos"]):
                        st.write(f"• {position}: {player}")
            
            with col_formation2:
                st.markdown(f"**{away_team} Formation:** {away_formation}")
                if away_lineup_data:
                    for player, position in zip(away_lineup_data["Unnamed: 0_level_0_Player"], away_lineup_data["Unnamed: 3_level_0_Pos"]):
                        st.write(f"• {position}: {player}")
            
            st.markdown("---")
            
            # Winner prediction
            col_a, col_b= st.columns(2)

            with col_a:
                st.metric(
                    label=f"📈 {home_team} Win Probability",
                    value=f"{round(prediction['home_win_prob']*100,2)}%"
                )
            
            with col_b:
                st.metric(
                    label=f"📈 {away_team} Win Probability",
                    value=f"{round(prediction['away_win_prob']*100,2)}%"
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
            
            st.markdown('</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()