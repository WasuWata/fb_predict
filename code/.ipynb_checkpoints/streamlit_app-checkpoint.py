import streamlit as st
import os
import pandas as pd
import numpy as np
import datetime
from datetime import date

# Set page configuration
st.set_page_config(
    page_title = 'Premier League Match Predictor',
    page_icon = '⚽',
    layout = 'wide'
)

# Custom CSS
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
    .stSelectbox > div > div {
        border: 2px solid #38003c;
    }
    </style>
    """, unsafe_allow_html=True)

premier_league_data = {
    "teams": [
        "Arsenal", "Manchester City", "Liverpool", "Chelsea", 
        "Manchester United", "Tottenham Hotspur", "Newcastle United",
        "Aston Villa", "West Ham United", "Brighton & Hove Albion",
        "Wolverhampton Wanderers", "Crystal Palace", "Everton",
        "Leicester City", "Leeds United", "Southampton", "Nottingham Forest",
        "Fulham", "Brentford", "Bournemouth"
    ],
    "players": {
        "Arsenal": ["Bukayo Saka", "Martin Ødegaard", "Gabriel Jesus", "Gabriel Martinelli", 
                   "Declan Rice", "William Saliba", "Aaron Ramsdale", "Kai Havertz"],
        "Manchester City": ["Erling Haaland", "Kevin De Bruyne", "Phil Foden", "Rodri", 
                          "Bernardo Silva", "John Stones", "Ederson", "Jack Grealish"],
        "Liverpool": ["Mohamed Salah", "Virgil van Dijk", "Alisson Becker", "Trent Alexander-Arnold",
                     "Darwin Núñez", "Luis Díaz", "Diogo Jota", "Dominik Szoboszlai"],
        "Chelsea": ["Raheem Sterling", "Enzo Fernández", "Thiago Silva", "Reece James",
                   "Cole Palmer", "Nicolas Jackson", "Moisés Caicedo", "Robert Sánchez"],
        "Manchester United": ["Bruno Fernandes", "Marcus Rashford", "Rasmus Højlund", "Casemiro",
                            "Harry Maguire", "André Onana", "Alejandro Garnacho", "Kobbie Mainoo"],
        # Add more teams and players as needed
    },
    "venues": ["Home", "Away", "Neutral"]
}

def create_mock_predictions():
    """Create mock historical prediction data"""
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='W')
    teams = premier_league_data["teams"][:6]
    
    predictions = []
    for _ in range(20):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        pred_date = np.random.choice(dates)
        
        predictions.append({
            'Date': pred_date,
            'Home Team': home_team,
            'Away Team': away_team,
            'Predicted Winner': np.random.choice([home_team, away_team, 'Draw']),
            'Confidence': np.random.uniform(50, 95)
        })
    
    return pd.DataFrame(predictions)

def predict_match(home_team, away_team, home_players, away_players, venue, form):
    """Mock prediction function - in real app, this would use a ML model"""
    # Simple mock prediction based on team names and form
    team_strength = {
        "Manchester City": 95, "Arsenal": 88, "Liverpool": 90, 
        "Chelsea": 85, "Manchester United": 83, "Tottenham Hotspur": 82,
        "Newcastle United": 80, "Aston Villa": 78, "West Ham United": 76,
        "Brighton & Hove Albion": 75, "Wolverhampton Wanderers": 72,
        "Crystal Palace": 70, "Everton": 68, "Leicester City": 66,
        "Leeds United": 65, "Southampton": 64, "Nottingham Forest": 63,
        "Fulham": 62, "Brentford": 61, "Bournemouth": 60
    }
    
    # Base strengths
    home_base = team_strength.get(home_team, 70)
    away_base = team_strength.get(away_team, 70)
    
    # Venue advantage
    if venue == "Home":
        home_base += 5
    elif venue == "Away":
        away_base += 5
    
    # Form factor
    form_bonus = {"Excellent": 5, "Good": 3, "Average": 0, "Poor": -3, "Terrible": -5}
    home_base += form_bonus.get(form['home'], 0)
    away_base += form_bonus.get(form['away'], 0)
    
    # Player count bonus (simple mock)
    home_base += len(home_players) * 0.5
    away_base += len(away_players) * 0.5
    
    # Calculate probabilities
    total = home_base + away_base
    home_win_prob = home_base / total * 100
    away_win_prob = away_base / total * 100
    draw_prob = 100 - (home_win_prob + away_win_prob) / 2
    
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
        'confidence': round(confidence, 1)
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Premier League Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict match outcomes using team and player data")
    
    # Sidebar for additional features
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        menu_option = st.radio(
            "Choose an option:",
            ["Match Prediction", "Historical Predictions", "Team Statistics"]
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_details = st.checkbox("Show detailed analysis", value=True)
        auto_update = st.checkbox("Auto-update predictions", value=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("This is a demonstration app for Premier League match predictions. Actual predictions would require a trained ML model with historical data.")
    
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
        
        if predict_button or auto_update:
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
            
            # Detailed analysis
            if show_details:
                st.markdown("### 📋 Match Analysis")
                
                analysis_points = [
                    f"• **Team Strength**: {home_team} vs {away_team}",
                    f"• **Venue Advantage**: {venue} venue selected",
                    f"• **Recent Form**: {home_team} ({home_form}) vs {away_team} ({away_form})",
                    f"• **Key Players Selected**: {len(home_players)} for {home_team}, {len(away_players)} for {away_team}",
                    f"• **Match Date**: {match_date.strftime('%B %d, %Y')}"
                ]
                
                for point in analysis_points:
                    st.write(point)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save prediction option
            if st.button("💾 Save This Prediction", type="secondary"):
                st.success(f"Prediction saved for {home_team} vs {away_team}")
    
    elif menu_option == "Historical Predictions":
        st.markdown("## 📜 Historical Predictions")
        
        # Show mock historical data
        predictions_df = create_mock_predictions()
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            team_filter = st.multiselect(
                "Filter by team",
                options=premier_league_data["teams"],
                default=[]
            )
        
        # Apply filters
        if team_filter:
            filtered_df = predictions_df[
                (predictions_df['Home Team'].isin(team_filter)) | 
                (predictions_df['Away Team'].isin(team_filter))
            ]
        else:
            filtered_df = predictions_df
        
        # Display table
        st.dataframe(
            filtered_df.sort_values('Date', ascending=False),
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn(format="MMM D, YYYY"),
                "Confidence": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
        
        # Statistics
        st.markdown("### 📈 Prediction Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(filtered_df))
        with col2:
            accuracy = np.random.uniform(65, 85)
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
        with col3:
            avg_conf = filtered_df['Confidence'].mean()
            st.metric("Average Confidence", f"{avg_conf:.1f}%")
    
    elif menu_option == "Team Statistics":
        st.markdown("## 📊 Team Statistics")
        
        selected_team = st.selectbox(
            "Select a team to view statistics",
            premier_league_data["teams"]
        )
        
        if selected_team:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {selected_team} Overview")
                st.write(f"**Key Players:**")
                players = premier_league_data["players"].get(selected_team, ["No data available"])
                for player in players[:5]:  # Show top 5 players
                    st.write(f"• {player}")
                
                st.write(f"\n**Team Strength Index:** {np.random.randint(70, 95)}")
                st.write(f"**Attack Rating:** {np.random.randint(70, 95)}")
                st.write(f"**Defense Rating:** {np.random.randint(70, 95)}")
            
            with col2:
                st.markdown("### Recent Performance")
                # Mock performance data
                performance_data = pd.DataFrame({
                    'Metric': ['Goals Scored', 'Goals Conceded', 'Clean Sheets', 'Avg Possession'],
                    'Value': [
                        np.random.randint(40, 80),
                        np.random.randint(20, 50),
                        np.random.randint(5, 20),
                        f"{np.random.randint(45, 65)}%"
                    ]
                })
                st.dataframe(performance_data, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📋 All Teams Comparison")
        
        # Create comparison table
        comparison_data = []
        for team in premier_league_data["teams"][:10]:  # First 10 teams for demo
            comparison_data.append({
                'Team': team,
                'Strength': np.random.randint(60, 95),
                'Attack': np.random.randint(60, 95),
                'Defense': np.random.randint(60, 95),
                'Form': np.random.choice(['📈', '📉', '➡️'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df.sort_values('Strength', ascending=False),
            use_container_width=True,
            column_config={
                "Form": st.column_config.TextColumn(
                    help="Trend: 📈 Improving, 📉 Declining, ➡️ Stable"
                )
            }
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Premier League Match Predictor | Demo Version | Data is simulated for demonstration purposes</p>
            <p>For accurate predictions, a trained machine learning model with historical data is required.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()