# Olympo Prediction

1. Install: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`

Keep `olympics_medal_table_2024.csv` and `olympic_best_model.pkl` in the same folder as `app.py`.

The left sidebar holds the navigation (Dataset Overview, Best Model, Total Medals Predictor,
Medal Forecast by Year, Report) plus the forecast assumption sliders. Use the arrow icon in the
top-left corner to collapse or reopen the sidebar.
Future-year forecasts use the growth sliders (percent change per Games), then the model predicts totals.
