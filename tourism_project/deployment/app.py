import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Jyotidarshan/tour-pckg-pred-model", filename="best_predictor_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tour Package acceptance prediction App")
st.write("This app preicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them based on their details.")
st.write("Kindly enter the customer details.")

# Collect user input
Age = st.number_input("Customer's Age", min_value=1)
TypeofContact = st.selectbox("Contact Type (How customer was contacted)", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier Type", ["1", "2", "3"])
DurationOfPitch = st.number_input("Sales Pitch Duration (in Minutes)")
Occupation = st.selectbox("Occupation Type", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Customer Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("No of person Accompanying customer")
NumberOfFollowups = st.number_input("No of followups made")
ProductPitched = st.selectbox("Pitched product", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Property Rating", ["3", "4", "5"])
MaritalStatus = st.selectbox("maritial Status", ["Married", "Divorced", "Unmarried", "Single"])
NumberOfTrips = st.number_input("Customer's avg trip count")
Passport = st.selectbox("Has passport?", ["0", "1"])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction score", ["1", "2", "3", "4", "5"])
OwnCar = st.number_input("Customer's car count")
NumberOfChildrenVisiting = st.number_input("No of Children Accompanying customer")
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Customer's Monthly Income", min_value=0.0)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age' : Age,
    'TypeofContact' : TypeofContact,
    'CityTier' : CityTier,
    'DurationOfPitch' : DurationOfPitch,
    'Occupation' : Occupation,
    'Gender' : Gender,
    'NumberOfPersonVisiting' : NumberOfPersonVisiting,
    'NumberOfFollowups' : NumberOfFollowups,
    'ProductPitched' : ProductPitched,
    'PreferredPropertyStar' : PreferredPropertyStar,
    'MaritalStatus' : MaritalStatus,
    'NumberOfTrips' : NumberOfTrips,
    'Passport' : Passport,
    'PitchSatisfactionScore' : PitchSatisfactionScore,
    'OwnCar' : OwnCar,
    'NumberOfChildrenVisiting' : NumberOfChildrenVisiting,
    'Designation' : Designation,
    'MonthlyIncome' : MonthlyIncome
}])

# Set the classification threshold
classification_threshold = 0.4

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Buy Tourism Package" if prediction == 1 else "NOT buy Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
