# GitHub + Streamlit Community Cloud Deployment

## 1. Create the GitHub repository

On GitHub, create a new repository named:

`smart-city-traffic-intelligence`

Recommended description:

`Interactive Smart City Traffic Analytics dashboard built with Python, Pandas, Plotly and Streamlit.`

Keep the repository public if you want recruiters to access the project easily.

## 2. Upload the project

Upload these items while preserving the folder structure:

- `app.py`
- `requirements.txt`
- `README.md`
- `INTERVIEW_NOTES.md`
- `.gitignore`
- `data/smart_city_traffic_cleaned.csv`

The important file for Streamlit is:

`app.py`

## 3. Deploy with Streamlit Community Cloud

Open Streamlit Community Cloud and sign in with GitHub.

Choose:

- Repository: `smart-city-traffic-intelligence`
- Branch: `main`
- Main file path: `app.py`

Click Deploy.

Streamlit will install the packages listed in `requirements.txt` and launch the application.

## 4. Test the live application

Check all five dashboard sections:

1. Executive Dashboard
2. Deep Dive
3. Anomalies
4. Congestion Prediction
5. Data Explorer

Also test the sidebar filters and the CSV download.

## 5. Resume/GitHub description

Suggested project title:

`Smart City Traffic Intelligence Dashboard`

Suggested resume bullet:

`Developed an interactive traffic analytics dashboard using Python, Pandas, Plotly and Streamlit to analyze 2,160 traffic observations across multiple junctions, with temporal analysis, congestion intelligence, anomaly detection and an ML-based classification extension.`

## Important

Do not commit passwords, API keys, tokens, or Streamlit secrets to GitHub.
