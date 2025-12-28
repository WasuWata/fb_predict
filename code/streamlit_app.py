# import streamlit as st
# import pandas as pd
# import numpy as np
# from datetime import date

# # premier league data
# url = 'https://raw.githubusercontent.com/WasuWata/fb_predict/main/code/source/summary.csv'
# data = pd.read_csv(url)
# teams = data['Team_Team'].unique().tolist()
# player_dict = {}
# for team in teams:
#     team_players = data[data['Team_Team'] == team]['Unnamed: 0_level_0_Player'].unique().tolist()
#     player_dict[team] = team_players

# premier_league_data = {}
# premier_league_data['teams'] = teams
# premier_league_data['players'] = player_dict
# premier_league_data['venues'] = ['Home','Away','Neutral']

# # Set page configuration
# st.set_page_config(
#     page_title="Premier League Match Predictor",
#     page_icon="⚽",
#     layout="wide"
# )

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #38003c;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         color: #00ff87;
#         font-size: 1.5rem;
#     }
#     .prediction-card {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #00ff87;
#         margin: 10px 0;
#     }
#     .team-card {
#         background-color: #ffffff;
#         padding: 15px;
#         border-radius: 8px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         margin: 10px 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# def predict_match(home_team, away_team, home_players, away_players, venue, form):
#     """Simple prediction function without ML dependencies"""
#     # Simple mock prediction
#     np.random.seed(hash(home_team + away_team) % 10000)
    
#     home_base = np.random.randint(60, 90)
#     away_base = np.random.randint(60, 90)
    
#     # Venue advantage
#     if venue == "Home":
#         home_base += 10
#     elif venue == "Away":
#         away_base += 5
    
#     # Form factor
#     form_bonus = {"Excellent": 10, "Good": 5, "Average": 0, "Poor": -5, "Terrible": -10}
#     home_base += form_bonus.get(form['home'], 0)
#     away_base += form_bonus.get(form['away'], 0)
    
#     # Calculate probabilities
#     total = home_base + away_base + 30  # Add 30 for draw possibility
#     home_win_prob = home_base / total * 100
#     away_win_prob = away_base / total * 100
#     draw_prob = 100 - home_win_prob - away_win_prob
    
#     # Determine winner
#     if home_win_prob > away_win_prob and home_win_prob > 35:
#         winner = home_team
#         confidence = home_win_prob
#     elif away_win_prob > home_win_prob and away_win_prob > 35:
#         winner = away_team
#         confidence = away_win_prob
#     else:
#         winner = "Draw"
#         confidence = draw_prob
    
#     return {
#         'home_win_prob': round(home_win_prob, 1),
#         'away_win_prob': round(away_win_prob, 1),
#         'draw_prob': round(draw_prob, 1),
#         'predicted_winner': winner,
#         'confidence': round(confidence, 1)
#     }

# def main():
#     # Header
#     st.markdown('<h1 class="main-header">⚽ Premier League Match Predictor</h1>', unsafe_allow_html=True)
#     st.markdown("### Predict match outcomes using team and player data")
    
#     # Sidebar
#     with st.sidebar:
#         st.markdown("### 📊 Navigation")
#         menu_option = st.radio(
#             "Choose an option:",
#             ["Match Prediction", "Historical Predictions", "Team Statistics"]
#         )
        
#         st.markdown("---")
#         st.markdown("### ⚙️ Settings")
#         show_details = st.checkbox("Show detailed analysis", value=True)
        
#         st.markdown("---")
#         st.markdown("### ℹ️ About")
#         st.info("This is a demonstration app for Premier League match predictions.")
    
#     if menu_option == "Match Prediction":
#         # Main content - Match prediction form
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown('<div class="team-card">', unsafe_allow_html=True)
#             st.markdown("### 🏠 Home Team")
#             home_team = st.selectbox(
#                 "Select Home Team",
#                 premier_league_data["teams"],
#                 key="home_team"
#             )
            
#             home_players = st.multiselect(
#                 f"Select Key Players for {home_team}",
#                 premier_league_data["players"].get(home_team, ["No player data available"]),
#                 key="home_players"
#             )
            
#             home_form = st.select_slider(
#                 f"{home_team} Recent Form",
#                 options=["Terrible", "Poor", "Average", "Good", "Excellent"],
#                 value="Average",
#                 key="home_form"
#             )
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         with col2:
#             st.markdown('<div class="team-card">', unsafe_allow_html=True)
#             st.markdown("### 🚗 Away Team")
#             away_team = st.selectbox(
#                 "Select Away Team",
#                 [team for team in premier_league_data["teams"] if team != home_team],
#                 key="away_team"
#             )
            
#             away_players = st.multiselect(
#                 f"Select Key Players for {away_team}",
#                 premier_league_data["players"].get(away_team, ["No player data available"]),
#                 key="away_players"
#             )
            
#             away_form = st.select_slider(
#                 f"{away_team} Recent Form",
#                 options=["Terrible", "Poor", "Average", "Good", "Excellent"],
#                 value="Average",
#                 key="away_form"
#             )
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         # Additional match details
#         st.markdown("---")
#         col3, col4 = st.columns(2)
        
#         with col3:
#             venue = st.selectbox(
#                 "Match Venue",
#                 premier_league_data["venues"],
#                 help="Select where the match is being played"
#             )
        
#         with col4:
#             match_date = st.date_input(
#                 "Match Date",
#                 value=date.today()
#             )
        
#         # Prediction button
#         st.markdown("---")
#         predict_button = st.button(
#             "🔮 Predict Match Outcome",
#             type="primary",
#             use_container_width=True
#         )
        
#         if predict_button:
#             # Get prediction
#             form_data = {'home': home_form, 'away': away_form}
#             prediction = predict_match(home_team, away_team, home_players, away_players, venue, form_data)
            
#             # Display prediction results
#             st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
#             st.markdown("## 📊 Prediction Results")
            
#             # Winner prediction
#             col_a, col_b, col_c = st.columns(3)
            
#             with col_a:
#                 st.metric(
#                     label=f"🏆 Predicted Winner",
#                     value=prediction['predicted_winner'],
#                     delta=f"{prediction['confidence']}% confidence"
#                 )
            
#             with col_b:
#                 st.metric(
#                     label=f"📈 {home_team} Win Probability",
#                     value=f"{prediction['home_win_prob']}%"
#                 )
            
#             with col_c:
#                 st.metric(
#                     label=f"📈 {away_team} Win Probability",
#                     value=f"{prediction['away_win_prob']}%"
#                 )
            
#             # Probability bars
#             st.markdown("### Win Probability Distribution")
#             prob_data = pd.DataFrame({
#                 'Outcome': [home_team, 'Draw', away_team],
#                 'Probability': [
#                     prediction['home_win_prob'], 
#                     prediction['draw_prob'], 
#                     prediction['away_win_prob']
#                 ]
#             })
#             st.bar_chart(prob_data.set_index('Outcome'))
            
#             st.markdown('</div>', unsafe_allow_html=True)
    
#     elif menu_option == "Historical Predictions":
#         st.markdown("## 📜 Historical Predictions")
#         st.info("Historical predictions feature would require database integration.")
        
#         # Create sample data
#         sample_data = pd.DataFrame({
#             'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
#             'Home Team': ['Arsenal', 'Liverpool', 'Chelsea', 'Man United', 'Man City'] * 2,
#             'Away Team': ['Tottenham', 'Everton', 'West Ham', 'Newcastle', 'Aston Villa'] * 2,
#             'Prediction': ['Home Win', 'Draw', 'Away Win', 'Home Win', 'Draw'] * 2,
#             'Accuracy': ['✓', '✓', '✗', '✓', '✓'] * 2
#         })
        
#         st.dataframe(sample_data, use_container_width=True)
    
#     elif menu_option == "Team Statistics":
#         st.markdown("## 📊 Team Statistics")
        
#         selected_team = st.selectbox(
#             "Select a team to view statistics",
#             premier_league_data["teams"]
#         )
        
#         if selected_team:
#             st.markdown(f"### {selected_team} Overview")
#             st.write(f"**Key Players:**")
#             players = premier_league_data["players"].get(selected_team, ["No data available"])
#             for player in players:
#                 st.write(f"• {player}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

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

# Set page configuration
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for better styling
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
    .formation-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        margin: 15px 0;
    }
    .position-box {
        background-color: white;
        border: 2px solid #00ff87;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        min-height: 80px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100px;
    }
    .position-label {
        font-weight: bold;
        color: #38003c;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .player-name {
        color: #495057;
        font-size: 0.8rem;
        word-wrap: break-word;
    }
    .formation-row {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    .gk-row {
        justify-content: center;
        margin-bottom: 30px;
    }
    .pitch-bg {
        background: linear-gradient(135deg, #00ff87 0%, #c1f1d8 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .clear-btn {
        background-color: #ff6b6b;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .clear-btn:hover {
        background-color: #ff5252;
    }
    </style>
    """, unsafe_allow_html=True)

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
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
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
                st.markdown('<div class="formation-row">', unsafe_allow_html=True)
                cols = st.columns(len(row_positions))
                
                for col_idx, (pos, unique_key, orig_idx) in enumerate(row_positions):
                    with cols[col_idx]:
                        create_position_box(pos, team_name, team_players, key_prefix, unique_key, orig_idx)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display selected lineup with clear buttons
    st.markdown("**Selected Lineup:**")
    
    # Create a container for the lineup with columns
    lineup_container = st.container()
    with lineup_container:
        # Group positions by type
        gks = []
        defenders = []
        midfielders = []
        forwards = []
        
        for unique_key, player in st.session_state[f'{key_prefix}_lineup'].items():
            if player:  # Only show filled positions
                pos_type = unique_key.split('_')[0]
                if pos_type == "GK":
                    gks.append((unique_key, player))
                elif pos_type in ["LB", "RB", "CB", "LWB", "RWB"]:
                    defenders.append((unique_key, player))
                elif pos_type in ["LM", "RM", "CM", "CDM", "CAM"]:
                    midfielders.append((unique_key, player))
                else:
                    forwards.append((unique_key, player))
        
        # Display in organized columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if gks:
                st.markdown("**Goalkeeper**")
                for unique_key, player in gks:
                    pos_label = unique_key.split('_')[0]
                    st.write(f"• {pos_label}: {player}")
                    if st.button("Clear", key=f"clear_{key_prefix}_{unique_key}", 
                               help=f"Clear {player} from {pos_label}"):
                        st.session_state[f'{key_prefix}_lineup'][unique_key] = ""
                        st.rerun()
        
        with col2:
            if defenders:
                st.markdown("**Defenders**")
                for unique_key, player in defenders:
                    pos_label = unique_key.split('_')[0]
                    st.write(f"• {pos_label}: {player}")
                    if st.button("Clear", key=f"clear_{key_prefix}_{unique_key}_def",
                               help=f"Clear {player} from {pos_label}"):
                        st.session_state[f'{key_prefix}_lineup'][unique_key] = ""
                        st.rerun()
        
        with col3:
            if midfielders:
                st.markdown("**Midfielders**")
                for unique_key, player in midfielders:
                    pos_label = unique_key.split('_')[0]
                    st.write(f"• {pos_label}: {player}")
                    if st.button("Clear", key=f"clear_{key_prefix}_{unique_key}_mid",
                               help=f"Clear {player} from {pos_label}"):
                        st.session_state[f'{key_prefix}_lineup'][unique_key] = ""
                        st.rerun()
        
        with col4:
            if forwards:
                st.markdown("**Forwards**")
                for unique_key, player in forwards:
                    pos_label = unique_key.split('_')[0]
                    st.write(f"• {pos_label}: {player}")
                    if st.button("Clear", key=f"clear_{key_prefix}_{unique_key}_fwd",
                               help=f"Clear {player} from {pos_label}"):
                        st.session_state[f'{key_prefix}_lineup'][unique_key] = ""
                        st.rerun()
        
        # Clear all button
        if st.button("Clear All Players", key=f"clear_all_{key_prefix}"):
            for unique_key in st.session_state[f'{key_prefix}_lineup']:
                st.session_state[f'{key_prefix}_lineup'][unique_key] = ""
            st.rerun()

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

def predict_match(home_team, away_team, home_players, away_players, venue, form):
    """Simple prediction function without ML dependencies"""
    # Simple mock prediction
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
    total = home_base + away_base + 30  # Add 30 for draw possibility
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
        'confidence': round(confidence, 1)
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
            ["Match Prediction", "Historical Predictions", "Team Statistics"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_details = st.checkbox("Show detailed analysis", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This is a demonstration app for Premier League match predictions.")
    
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
                
                home_form = st.select_slider(
                    f"{home_team} Recent Form",
                    options=["Terrible", "Poor", "Average", "Good", "Excellent"],
                    value="Average",
                    key="home_form"
                )
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
                
                away_form = st.select_slider(
                    f"{away_team} Recent Form",
                    options=["Terrible", "Poor", "Average", "Good", "Excellent"],
                    value="Average",
                    key="away_form"
                )
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
            # Get selected players from lineups
            home_lineup = st.session_state.get('home_lineup', {})
            away_lineup = st.session_state.get('away_lineup', {})
            
            home_selected_players = [player for player in home_lineup.values() if player]
            away_selected_players = [player for player in away_lineup.values() if player]
            
            # Get prediction
            form_data = {'home': home_form, 'away': away_form}
            prediction = predict_match(home_team, away_team, home_selected_players, away_selected_players, venue, form_data)
            
            # Display prediction results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("## 📊 Prediction Results")
            
            # Display formations
            col_formation1, col_formation2 = st.columns(2)
            with col_formation1:
                st.markdown(f"**{home_team} Formation:** {home_formation}")
                for unique_key, player in home_lineup.items():
                    if player:
                        pos = unique_key.split('_')[0]
                        st.write(f"• {pos}: {player}")
            
            with col_formation2:
                st.markdown(f"**{away_team} Formation:** {away_formation}")
                for unique_key, player in away_lineup.items():
                    if player:
                        pos = unique_key.split('_')[0]
                        st.write(f"• {pos}: {player}")
            
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
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif menu_option == "Historical Predictions":
        st.markdown("## 📜 Historical Predictions")
        st.info("Historical predictions feature would require database integration.")
        
        # Create sample data
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
            'Home Team': ['Arsenal', 'Liverpool', 'Chelsea', 'Man United', 'Man City'] * 2,
            'Away Team': ['Tottenham', 'Everton', 'West Ham', 'Newcastle', 'Aston Villa'] * 2,
            'Prediction': ['Home Win', 'Draw', 'Away Win', 'Home Win', 'Draw'] * 2,
            'Accuracy': ['✓', '✓', '✗', '✓', '✓'] * 2
        })
        
        st.dataframe(sample_data, use_container_width=True)
    
    elif menu_option == "Team Statistics":
        st.markdown("## 📊 Team Statistics")
        
        selected_team = st.selectbox(
            "Select a team to view statistics",
            premier_league_data["teams"]
        )
        
        if selected_team:
            st.markdown(f"### {selected_team} Overview")
            st.write(f"**Available Players:**")
            players = premier_league_data["players"].get(selected_team, ["No data available"])
            for player in players:
                st.write(f"• {player}")
            
            # Show formation options
            st.markdown(f"**Common Formations for {selected_team}:**")
            for formation_name, formation_data in formations.items():
                st.write(f"• {formation_name}: {formation_data['defenders']}-{formation_data['midfielders']}-{formation_data['forwards']}")

if __name__ == "__main__":
    main()