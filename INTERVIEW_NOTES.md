# Interview Notes

## 60-second explanation
I developed a Smart City Traffic Analytics and Congestion Intelligence Dashboard using Python. I started with a CSV traffic dataset and performed data validation, cleaning and feature engineering. I derived time-based features such as day type and time period, then performed EDA to understand traffic patterns across hours and junctions. I used Pandas for analysis and Plotly with Streamlit for an interactive dashboard. I also added IQR-based anomaly analysis and a Random Forest model as an optional congestion-classification layer.

## Important result
The linear correlation between vehicle count and average speed is approximately 0.004 in this dataset. I therefore avoided claiming that increased traffic volume directly causes lower speed and instead analyzed congestion across time and locations.

## ML result
Development hold-out accuracy: 100.00%.
Use this only as a dataset-specific test metric, not as a general deployment claim.

## Why no SQL?
This version intentionally uses a file-based analytics pipeline so the project focuses on the analyst workflow: ingestion, cleaning, transformation, analysis, visualization and interpretation. SQL/database integration can be added when the data source requires it, but it is not necessary for this analytics demonstration.
