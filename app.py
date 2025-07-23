import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- HEADERS ----------
st.markdown("""
<div style='text-align: center;'>
    <h1 style='font-size: 3rem; text-transform: uppercase; text-decoration: underline;'>💼 Employee Salary Prediction</h1>
</div>
<div style='text-align: center;'>
    <h4 style='font-size: 1.5rem; text-decoration: underline;'>🔍 U s i n g  L i n e a r  R e g r e s s i o n</h4>
</div>

""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")
df = pd.read_csv("final_salary_data.csv")

X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")
y_pred = joblib.load("y_pred.pkl")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------- LAYOUT ----------
cols = st.columns([1, 2])
input_area = cols[0]
plot_area = cols[1]

# ---------- INPUT SECTION ----------
with input_area:
    st.subheader("👤 Input Details")
    education = st.selectbox("Education", label_encoders['Education'].classes_)
    experience = st.slider("Experience", 0, 30, 5)
    job_title = st.selectbox("Job Title", label_encoders['Job_Title'].classes_)
    age = st.slider("Age", 18, 70, 30)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    location = st.selectbox("Location", label_encoders['Location'].classes_)

    input_df = pd.DataFrame([{
        'Education': education,
        'Experience': experience,
        'Job_Title': job_title,
        'Age': age,
        'Gender': gender,
        'Location': location
    }])[feature_columns]

    for col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

    if st.button("PREDICT MY SALARY"):
        salary = model.predict(input_df)[0]
        st.markdown(f"""
        <div style='text-align: center;'>
            <h2 style='color: green; animation: fadeInScale 1s ease forwards;'>💵 Estimated Salary: ${salary:,.2f}</h2>
        </div>
        <style>
        @keyframes fadeInScale {{
            0% {{ opacity: 0; transform: scale(0.8); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        </style>
        """, unsafe_allow_html=True)

# ---------- DATASET & MODEL PERFORMANCE ----------
with plot_area:
    st.subheader("📂 Sample of Dataset")
    st.dataframe(df.head(20))

    # ✅ Styled Model Performance Card
    st.markdown("""
    <div style='background-color: #112B3C; padding: 2rem; border-radius: 10px;'>
        <h3 style='color: white;'>📊 Model Performance & Plots Graph</h3>
        <div style='display: flex; justify-content: space-around;'>
            <div style='text-align: center;'>
                <p style='font-weight: bold; color: #ffffff;'>Mean Absolute Error (MAE)</p>
                <h2 style='color: #00FFAA;'>${:,.2f}</h2>
            </div>
            <div style='text-align: center;'>
                <p style='font-weight: bold; color: #ffffff;'>Root Mean Squared Error (RMSE)</p>
                <h2 style='color: #00FFAA;'>${:,.2f}</h2>
            </div>
            <div style='text-align: center;'>
                <p style='font-weight: bold; color: #ffffff;'>R-squared (R2) Score</p>
                <h2 style='color: #00FFAA;'>{:.4f}</h2>
            </div>
        </div>
    </div>
    """.format(mae, rmse, r2), unsafe_allow_html=True)

    # ---------- EXPANDABLE PLOTS ----------
    with st.expander("📈 Actual vs Predicted Scatter Plot"):
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1, color='#0f3460')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='gray')
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("📈 Actual vs Predicted")
        st.pyplot(fig1)

    with st.expander("📉 Residuals Distribution"):
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, kde=True, ax=ax2, color='#e94560')
        ax2.set_title("📉 Residuals Distribution")
        st.pyplot(fig2)

    with st.expander("🔍 Experience vs Salary Scatter Plot"):
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df['Experience'], y=df['Salary_USD'], ax=ax3, color='purple')
        ax3.set_title("🔍 Experience vs Salary")
        ax3.set_xlabel("Experience")
        ax3.set_ylabel("Salary (USD)")
        st.pyplot(fig3)

    with st.expander("📊 Education Count Bar Plot"):
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.countplot(y=df['Education'], ax=ax4, palette='viridis')
        ax4.set_title("📊 Education Count")
        ax4.set_xlabel("Count")
        ax4.set_ylabel("Education Level")
        st.pyplot(fig4)
