    
import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.express as px
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model("D:\Data Science Class\project\Stock prediction\stock price prediction model.keras")

# Set title and sidebar
st.title('Stock Price Prediction')
st.sidebar.header('User Input')

# User input for stock name
stock_name = st.sidebar.text_input(label='Enter Stock Name', value='AAPL')

# Fetch data from Yahoo Finance
try:
    df = yf.download(stock_name)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display live chart from Yahoo Finance
st.subheader('Live Chart from Yahoo Finance')
fig = px.line(df, title="Stock Live Price", x=df.index, y=df['Close'],
              labels={'x': 'Date', 'y': 'Close price USD ($)'},
              line_shape='spline', render_mode='svg', 
              template='plotly_dark', width=800, height=400)
st.plotly_chart(fig)

# Display stock data
st.subheader('Stock Data')
st.write(df.tail(5))

# Visualize the closing price history
st.subheader('Closing Price History')
fig1 = plt.figure(figsize=(10, 6))
plt.plot(df['Close'], color='skyblue')
plt.title('Close Price History', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close price USD ($)', fontsize=14)
st.pyplot(fig1)

# Preprocess data for model training
data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Train-test split
train_data = scaled_data[0:training_data_len, :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Testing data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Model predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Model evaluation
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Reshape valid['Close'] to match the shape of predictions
valid_close = valid['Close'].values.reshape(-1, 1)




# Plotting predictions
st.subheader('Validation vs. Predictions')
fig2 = plt.figure(figsize=(10, 6))
plt.plot(data['Close'], color='blue', label='Train')
plt.plot(valid['Close'], color='green', label='Validation')
plt.plot(valid['Predictions'], color='red', label='Predictions')
plt.title('Model', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close price USD ($)', fontsize=14)
plt.legend()
st.pyplot(fig2)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - valid_close) ** 2))

st.subheader('Model Evaluation')
st.write(f'Root Mean Squared Error (RMSE): {rmse}')

# Dynamic prediction
user_prediction_date = st.sidebar.date_input("Select a date for prediction", datetime.today())

# Function to make dynamic predictions
def make_dynamic_prediction(stock_data, model, scaler, user_date):
    user_date = pd.to_datetime(user_date)
    user_data = stock_data[stock_data.index <= user_date][['Close']].copy()
    scaled_data = scaler.transform(user_data.values)
    x_input = []
    x_input.append(scaled_data[-60:])
    x_input = np.array(x_input)
    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))
    predicted_price_scaled = model.predict(x_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    return predicted_price.item()

# Make dynamic prediction based on user input
predicted_price = make_dynamic_prediction(data, model, scaler, user_prediction_date)

# Display prediction
st.subheader('Prediction')
if st.button(label="Show prediction", key="btn"):
    
    predicted_price_scalar = predicted_price

    rounded_price = round(predicted_price_scalar, 2)
    st.write("The predicted stock price for Apple Inc. is $", rounded_price, "USD.")
