import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# premier league data
url = 'https://raw.githubusercontent.com/WasuWata/fb_predict/main/code/source/summary.csv'
data = pd.read_csv(url)
teams = data['Team_Team'].unique().tolist()
player_dict = {}
for team in teams:
    team_players = data[data['Team_Team'] == team]['Unnamed: 0_level_0_Player'].unique().tolist()
    player_dict[team] = team_players

premier_league_data = {}
premier_league_data['Team'] = teams
premier_league_data['players'] = player_dict
premier_league_data['venues'] = ['Home','Away','Neutral']

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
    </style>
    """, unsafe_allow_html=True)

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
    st.markdown("### Predict match outcomes using team and player data")
    
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
            
            home_players = st.multiselect(
                f"Select Key Players for {home_team}",
                premier_league_data["players"].get(home_team, ["No player data available"]),
                key="home_players"
            )
            
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
            
            away_players = st.multiselect(
                f"Select Key Players for {away_team}",
                premier_league_data["players"].get(away_team, ["No player data available"]),
                key="away_players"
            )
            
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
            # Get prediction
            form_data = {'home': home_form, 'away': away_form}
            prediction = predict_match(home_team, away_team, home_players, away_players, venue, form_data)
            
            # Display prediction results
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown("## 📊 Prediction Results")
            
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
            st.write(f"**Key Players:**")
            players = premier_league_data["players"].get(selected_team, ["No data available"])
            for player in players:
                st.write(f"• {player}")

if __name__ == "__main__":
    main()