import pandas as pd
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import math 
app = Flask(__name__)

# ---------------------- Load Datasets ---------------------- #
patient_data = pd.read_csv("final_icu_scored_with_label.csv")

try:
    patients_info = pd.read_csv("patients.csv")
    patient_data = pd.merge(patient_data, patients_info[['Id', 'FIRST']], left_on='PATIENT', right_on='Id', how='left')
    patient_data.rename(columns={"FIRST": "FIRST_NAME"}, inplace=True)
    patient_data.drop(columns=["Id"], inplace=True)
except:
    patient_data['FIRST_NAME'] = 'Unknown'

encounters = pd.read_csv("encounters.csv", parse_dates=['START'])
medications = pd.read_csv("medications.csv", parse_dates=['START'])
procedures = pd.read_csv("procedures.csv", parse_dates=['START'])
conditions = pd.read_csv("conditions.csv", parse_dates=['START'])

# ---------------------- ICU Setup ---------------------- #
encounters['date'] = encounters['START'].dt.date
icu_encounters = encounters[encounters['ENCOUNTERCLASS'].str.lower().isin(['inpatient', 'emergency'])].copy()
icu_trend_data = icu_encounters[['PATIENT', 'date']].copy()
icu_trend_data['day_of_week'] = pd.to_datetime(icu_trend_data['date']).dt.day_name()

# ---------------------- Forecasting ---------------------- #
icu_beds = icu_encounters.groupby('date').size().reset_index(name='icu_beds')
ventilators = procedures.groupby(procedures['START'].dt.date).size().reset_index(name='ventilators').rename(columns={"START": "date"})
medications_usage = medications.groupby(medications['START'].dt.date)['TOTALCOST'].sum().reset_index(name='medications').rename(columns={"START": "date"})

resource_forecast = pd.merge(icu_beds, ventilators, how='outer', on='date')
resource_forecast = pd.merge(resource_forecast, medications_usage, how='outer', on='date')
resource_forecast = resource_forecast.sort_values('date').fillna(0)

# ---------------------- Final ICU Condition Groups ---------------------- #
condition_groups = {
    "Respiratory Disorders": ["acute bronchitis (disorder)", "acute viral pharyngitis (disorder)", "viral sinusitis (disorder)"],
    "Chronic Health Issues": ["body mass index 30+ - obesity (finding)", "anemia (disorder)"],
    "Psychosocial Factors": ["stress (finding)", "social isolation (finding)"]
}

# ---------------------- Routes ---------------------- #
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/icu_trends', methods=['GET'])
def icu_trends():
    try:
        selected_days = int(request.args.get('days', 365))
    except:
        selected_days = 365

    selected_group = request.args.get('group', 'All') or 'All'

    start_date = datetime.now().date() - timedelta(days=selected_days)
    filtered_encounters = icu_encounters[icu_encounters['date'] >= start_date]

    lowercase_group_map = {k.lower(): v for k, v in condition_groups.items()}

    if selected_group.lower() != 'all':
        keywords = lowercase_group_map.get(selected_group.lower(), [])
        matched_conditions = conditions[
            conditions['DESCRIPTION'].str.lower().isin([k.lower() for k in keywords])
        ]
        matched_patients = matched_conditions[['PATIENT']].drop_duplicates()
        filtered_encounters = pd.merge(filtered_encounters, matched_patients, on='PATIENT')

    if filtered_encounters.empty:
        return render_template('icu_trends.html', trend_data=False, selected_days=selected_days, selected_group=selected_group)

    trends = (
        filtered_encounters.groupby('date')
        .size()
        .reset_index(name='admissions')
    )
    trends['date'] = pd.to_datetime(trends['date'])

    labels = trends['date'].dt.strftime("%Y-%m-%d").tolist()
    values = trends['admissions'].tolist()

    return render_template(
        'icu_trends.html',
        trend_data=True,
        labels=labels,
        values=values,
        selected_days=selected_days,
        selected_group=selected_group,
        total_patients=filtered_encounters['PATIENT'].nunique()
    )

@app.route('/risk_classification')
def risk_classification():
    return render_template('risk_classification.html', patients=patient_data.to_dict(orient="records"))

@app.route('/resource_forecasting')
def resource_forecasting():
    return render_template('resource_forecasting.html')

# ---------------------- APIs ---------------------- #
@app.route('/api/get_risk_data', methods=['GET'])
def get_risk_data():
    try:
        display_df = patient_data.copy()

        # Map ML-based columns properly
        if 'Risk_Score_ML' in display_df.columns:
            display_df.rename(columns={
                'Risk_Score_ML': 'Risk_Score',
                'Final_Risk_Category_ML': 'Final_Risk_Category'
            }, inplace=True)

        required_cols = ['PATIENT', 'FIRST_NAME', 'AGE', 'Risk_Score', 'Final_Risk_Category']
        display_df = display_df[required_cols]

        return jsonify(display_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/get_icu_trends', methods=['GET'])
def get_icu_trends():
    days = int(request.args.get('days', 365))
    condition = request.args.get('condition', 'All').lower()

    start_date = datetime.now().date() - timedelta(days=days)
    filtered_encounters = icu_encounters[icu_encounters['date'] >= start_date]

    lowercase_group_map = {k.lower(): v for k, v in condition_groups.items()}

    if condition != 'all':
        keywords = lowercase_group_map.get(condition, [])
        matched_conditions = conditions[
            conditions['DESCRIPTION'].str.lower().isin([k.lower() for k in keywords])
        ]
        matched_patients = matched_conditions[['PATIENT']].drop_duplicates()
        filtered_encounters = pd.merge(filtered_encounters, matched_patients, on='PATIENT')

    if filtered_encounters.empty:
        return jsonify({
            "trends": [],
            "patient_count": 0
        })

    trends = (
        filtered_encounters.groupby('date')
        .size()
        .reset_index(name='admissions')
    )
    trends['day_of_week'] = pd.to_datetime(trends['date']).dt.day_name()

    return jsonify({
        "trends": trends.to_dict(orient="records"),
        "patient_count": filtered_encounters['PATIENT'].nunique()
    })

@app.route('/api/get_resource_forecast', methods=['GET'])
def get_resource_forecast():
    try:
        print("📦 Reading patients.csv...")
        patients = pd.read_csv("cleaned/patients.csv")

        print("📅 Parsing ENCOUNTER_DATE...")
        patients['ENCOUNTER_DATE'] = pd.to_datetime(patients['ENCOUNTER_DATE'], errors='coerce')
        patients = patients.dropna(subset=['ENCOUNTER_DATE'])
        patients['ENCOUNTER_DATE'] = patients['ENCOUNTER_DATE'].dt.date

        if 'ICU_Demand_ML' not in patients.columns:
            raise Exception("❌ Column ICU_Demand_ML not found in dataset.")

        print("📊 Grouping daily demand...")
        daily_demand = patients.groupby('ENCOUNTER_DATE')['ICU_Demand_ML'].sum().reset_index()
        daily_demand.rename(columns={'ICU_Demand_ML': 'Total_ICU_Demand'}, inplace=True)
        daily_demand['Beds_Required'] = daily_demand['Total_ICU_Demand'].apply(lambda x: math.ceil(x / 10))

        print("📆 Applying forecast range filter...")
        range_days = int(request.args.get("range", 30))
        latest_date = max(daily_demand['ENCOUNTER_DATE'])
        start_date = latest_date - timedelta(days=range_days)

        print(f"➡️ Latest Date: {latest_date}, Range: {range_days}, Start: {start_date}")
        filtered = daily_demand[daily_demand['ENCOUNTER_DATE'] >= start_date].copy()

        if filtered.empty:
            raise Exception("⚠️ No data found for the selected forecast range.")

        print("✅ Forecast data prepared.")
        filtered['ENCOUNTER_DATE'] = filtered['ENCOUNTER_DATE'].astype(str)
        return jsonify(filtered.rename(columns={"ENCOUNTER_DATE": "date"}).to_dict(orient="records"))

    except Exception as e:
        print("🔴 Forecasting API error:", str(e))
        return jsonify({"error": str(e)})


@app.route('/api/search_patient', methods=['GET'])
def search_patient():
    query = request.args.get("query", "").lower()
    result = patient_data[
        patient_data['PATIENT'].str.lower().str.contains(query) |
        patient_data['FIRST_NAME'].str.lower().str.contains(query)
    ].copy()
    result = result.rename(columns={
        'Risk_Score_ML': 'Risk_Score',
        'Final_Risk_Category_ML': 'Final_Risk_Category'
    })
    required_cols = ['PATIENT', 'FIRST_NAME', 'AGE', 'Risk_Score', 'Final_Risk_Category']
    result = result[required_cols]
    return jsonify(result.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
