# ICU Risk Prediction and Resource Forecasting System

This is a complete machine learning and data engineering pipeline that predicts ICU risk, classifies patients by risk category, and forecasts ICU resource needs using synthetic EHR data.

Built using **Python**, **Flask**, and **scikit-learn**, this system simulates a real-world ICU triage tool for prioritizing care and planning resource allocation.

---

## Project Goals

- Predict ICU admission probability
- Classify patients as Low / Medium / High risk
- Calculate Base Resource Unit (BRU) to measure resource demand
- Forecast ICU bed and cost demand for the next 7–30 days
- Visualize insights using a Flask GUI

---

## Technologies Used

- **Python 3.10**
- **Flask** – Web application
- **Pandas, NumPy** – Data wrangling
- **Matplotlib, Seaborn** – EDA and visualization
- **Scikit-learn** – Modeling (Linear & Logistic Regression)
- **Docker** – Containerization

---

## Machine Learning Models

| Task                      | Model Used           | Target            |
|---------------------------|----------------------|-------------------|
| Risk Score Prediction     | Linear Regression     | `Final_Risk_Score` (continuous) |
| ICU Admission Prediction  | Logistic Regression   | `ICU_ADMISSION` (binary) |
| BRU Classification        | Logistic Regression   | `BRU_Category` (binary) |

> Models were selected based on target type and interpreted using clinical logic.

---

## Risk Score Formula (Manually Weighted)

\[
\text{Risk Score} = w_1A + w_2V + w_3L + w_4C + w_5M + w_6E
\]

Where:
- `A`: Normalized Age  
- `V`: Vital Signs Score  
- `L`: Lab Results Score  
- `C`: Comorbidity Score  
- `M`: Medications & Procedures Score  
- `E`: Encounter Score

> Weights were based on clinical references like **NEWS2**, **Charlson Index**, and ICU triage protocols.

---

## Base Resource Unit (BRU) Formula

\[
BRU = \alpha_1 \cdot (\text{Base Encounter Cost} + \text{Procedure Cost} + \text{Medication Cost}) + \alpha_2 \cdot (\beta_1 \cdot \text{Emergency Visits})
\]

> Designed to reflect cost burden and urgency. Used for forecasting ICU resource load.

---

## Data Sources

Data was generated using **[Synthea](https://synthetichealth.github.io/synthea/)** – a tool that simulates realistic but synthetic electronic health records.

### Files used:

- `patients.csv`  
- `observations.csv`  
- `conditions.csv`  
- `medications.csv`  
- `procedures.csv`  
- `encounters.csv`

All datasets are hosted externally due to GitHub size limits:  
🔗 [Access via Google Drive]https://drive.google.com/drive/folders/1fzN-hZqPjJudntMW8kggDMrAHszBdNim?usp=drive_link

---

## Pipeline Overview

1. **Data Cleaning**  
   Removed deceased and age > 100  
2. **Feature Engineering**  
   Aggregated vitals, labs, encounters, and conditions  
3. **Manual Scoring**  
   Computed Risk Score and BRU using clinically weighted formulas  
4. **ML Modeling**  
   Linear & Logistic regression for predictions  
5. **Visualization**  
   Flask-based web interface for interaction

---

## Run Locally

```bash
git clone https://github.com/siva-279/icu-risk-prediction.git
cd icu-risk-prediction
pip install -r requirements.txt
python app.py
