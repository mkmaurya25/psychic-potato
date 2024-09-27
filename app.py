import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

# Function to get user input
def get_user_input():
    st.title("Car Price Prediction App")

    # Input fields
    year = st.number_input('Year', min_value=1900, max_value=2023, value=2020)
    present_price = st.number_input('Present Price', min_value=0.0, value=0.0)
    kms_driven = st.number_input('Kms Driven', min_value=0, value=0)
    owner = st.selectbox('Owner', [0, 1, 2])  # Adjust this based on your dataset
    fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
    seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer'])
    transmission_type = st.selectbox('Transmission Type', ['Manual', 'Automatic'])

    return year, present_price, kms_driven, owner, fuel_type, seller_type, transmission_type

# Function to make prediction
def predict_price(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission_type):
    # Preprocessing the input data
    kms_driven_log = np.log(kms_driven) if kms_driven > 0 else 0  # Handle log(0) case

    # Encoding categorical variables
    fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
    fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0
    year = 2020 - year
    seller_type_individual = 1 if seller_type == 'Individual' else 0
    transmission_manual = 1 if transmission_type == 'Manual' else 0

    # Making prediction
    prediction = model.predict([[present_price, kms_driven_log, owner, year, fuel_type_diesel, fuel_type_petrol, seller_type_individual, transmission_manual]])
    return round(prediction[0], 2)

# Streamlit app main logic
if __name__ == '__main__':
    year, present_price, kms_driven, owner, fuel_type, seller_type, transmission_type = get_user_input()

    if st.button('Predict Price'):
        output = predict_price(year, present_price, kms_driven, owner, fuel_type, seller_type, transmission_type)

        if output < 0:
            st.error("Sorry, you cannot sell this car.")
        else:
            st.success(f"You can sell the car at: ₹{output}")
