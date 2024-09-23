import streamlit as st
import numpy as np
from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv('PreparedMenstruationCycleDataset.csv')

y = data["LengthofCycle"]
x = data[['EstimatedDayofOvulation', 'LengthofLutealPhase', 'LengthofMenses', 
           'MensesScoreDayOne', 'MensesScoreDayTwo', 'MensesScoreDayThree', 
           'MensesScoreDayFour', 'MensesScoreDayFive', 'TotalMensesScore', 
           'NumberofDaysofIntercourse', 'UnusualBleeding']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.8, random_state=42)

random_forest = RandomForestRegressor(n_estimators=49, criterion="squared_error", max_depth=67)
random_forest.fit(x_train, y_train)

def predict_cycle_length(inputs):
    inputs_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
    predicted_length = random_forest.predict(inputs_scaled)
    return int(predicted_length[0])  

st.title("Vorhersage der Länge des Menstruationszyklus")

st.info("""
**Wie funktionierts?**
1. Fülle alle Felder zu deinem letzten Zyklus aus (Geschätzter Tag des Eisprungs, Länge der Lutealphase, Dauer der letzten Menstruationsblutung, Anzahl der Tage mit Geschlechtsverkehr, Ungewöhnliche Blutungen).
2. Bewerte die täglichen Blutungsstärke deiner Menstruation des letzten Zyklus auf einer Skala von 0 (keine Blutung) bis 10 (sehr starke Blutung).
3. Gebe das Datum deiner letzten Menstruationsblutung an und drücke den Button „Vorhersage der Zykluslänge anzeigen“.
4. Basierend auf diesen Eingaben berechnet die App die voraussichtliche Länge deines nächsten Zyklus und somit auch das Datum deines nächsten Mentrusationsbeginns.
""")
st.header("Eingabefelder für die Vorhersage")

estimated_day_of_ovulation = st.number_input('Geschätzter Tag des Eisprungs', min_value=1, max_value=30, value=14)
length_of_luteal_phase = st.number_input('Länge der Lutealphase (die Zeit nach dem Eisprung bis zum Beginn der nächsten Menstruation), in Tage', min_value=1, max_value=30, value=14)
length_of_menses = st.number_input('Dauer der letzten Menstruationsblutung, in Tage', min_value=0, max_value=7, value=5)

menses_score_day_1 = 1
menses_score_day_2 = 0
menses_score_day_3 = 0
menses_score_day_4 = 0
menses_score_day_5 = 0

if length_of_menses < 5:
    for i in range(1, 6):
        if i > length_of_menses:
            st.slider(f'Blutungsstärke Tag {i}', min_value=0, max_value=10, value=0, disabled=True)
        else:
            if i == 1:
                menses_score_day_1 = st.slider(f'Blutungsstärke Tag {i}', min_value=0, max_value=10, value=0)
            elif i == 2:
                menses_score_day_2 = st.slider(f'Blutungsstärke Tag {i}', min_value=0, max_value=10, value=0)
            elif i == 3:
                menses_score_day_3 = st.slider(f'Blutungsstärke Tag {i}', min_value=0, max_value=10, value=0)
            elif i == 4:
                menses_score_day_4 = st.slider(f'Blutungsstärke Tag {i}', min_value=0, max_value=10, value=0)
else:
    menses_score_day_1 = st.slider('Blutungsstärke Tag 1', min_value=0, max_value=10, value=0)
    menses_score_day_2 = st.slider('Blutungsstärke Tag 2', min_value=0, max_value=10, value=0)
    menses_score_day_3 = st.slider('Blutungsstärke Tag 3', min_value=0, max_value=10, value=0)
    menses_score_day_4 = st.slider('Blutungsstärke Tag 4', min_value=0, max_value=10, value=0)
    menses_score_day_5 = st.slider('Blutungsstärke Tag 5', min_value=0, max_value=10, value=0)

number_of_days_of_intercourse = st.number_input('Anzahl der Tage mit Geschlechtsverkehr', min_value=0, max_value=31, value=0)

unusual_bleeding = st.selectbox('Ungewöhnliche Blutungen?', options=[0, 1], format_func=lambda x: 'Ja' if x == 1 else 'Nein')

total_menses_score = menses_score_day_1 + menses_score_day_2 + menses_score_day_3 + menses_score_day_4 + menses_score_day_5

st.write(f"Summe der Blutungsstärke: **{total_menses_score}**")

last_menses_start_date = st.date_input('Datum des Beginns der letzten Menstruation')

if st.button('Vorhersage der Zykluslänge anzeigen'):
    inputs = [
        estimated_day_of_ovulation,
        length_of_luteal_phase,
        length_of_menses,
        menses_score_day_1,
        menses_score_day_2,
        menses_score_day_3,
        menses_score_day_4,
        menses_score_day_5,
        total_menses_score,
        number_of_days_of_intercourse,
        unusual_bleeding
    ]
    
    predicted_cycle_length = predict_cycle_length(inputs)

    st.success(f"Die vorhergesagte Länge deines Menstruationszyklus beträgt **{predicted_cycle_length} Tage**.")
        
    next_menses_start_date = last_menses_start_date + timedelta(days=predicted_cycle_length)
    weekday_name = next_menses_start_date.strftime('%A')
    formatted_date = next_menses_start_date.strftime('%d.%m.%Y')  
    
    st.write(f"Das voraussichtliche Datum des nächsten Menstruationsbeginns ist **{weekday_name}, {formatted_date}**.")