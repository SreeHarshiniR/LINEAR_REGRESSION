import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# App title
st.title("📊 Simple Data Science Streamlit App")

st.write("This app demonstrates a basic ML model using Streamlit")

# Sample dataset
data = {
    "Hours_Studied": [1, 2, 3, 4, 5],
    "Score": [35, 40, 50, 65, 75]
}
df = pd.DataFrame(data)

st.subheader("📁 Dataset")
st.dataframe(df)

# Model training
X = df[["Hours_Studied"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)

# User input
st.subheader("🔢 Enter Study Hours")
hour = st.number_input("Hours Studied", min_value=0.0, step=0.5)

# Prediction
if st.button("Predict Score"):
    input_data = pd.DataFrame([[hour]], columns=["Hours_Studied"])
    prediction = model.predict(input_data)
    st.success(f"Predicted Score: {prediction[0]:.2f}")
