import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import openai
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image as PILImage

# Function to split time series data for LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Load data and preprocess
def load_data(uploaded_file):
    data = pd.read_excel(uploaded_file)
    data = data.rename(columns={'Month': 'ds', 'Actual POs': 'y'})
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce', format='%B')
    data = data.dropna(subset=['ds', 'y'])
    data['ds'] = data['ds'].apply(lambda x: x.replace(year=2024))
    return data.dropna()

# Define ML model training functions
def train_arima(data):
    model = auto_arima(data, seasonal=False, trace=False)
    predictions = model.predict_in_sample()
    rmse = np.sqrt(mean_squared_error(data, predictions))
    return model, rmse

def train_lstm(data, n_steps):
    X, y = create_sequences(data, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.fit(X, y, epochs=50, verbose=0)
    try:
        predictions = model.predict(X)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return model, rmse

def train_linear_regression(data):
    X = data.index.values.reshape(-1, 1)
    y = data.values
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return model, rmse

def train_random_forest(data):
    X = data.index.values.reshape(-1, 1)
    y = data.values
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return model, rmse

# Create PDF report
def create_pdf_report(data, models, best_model_name, forecast, fig_path, uploaded_file_name):
    filename = 'Time_Series_Forecast_Report.pdf'
    document = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()

    # Title
    document.setTitle("Forecasting Report")
    document.setFont("Helvetica-Bold", 18)
    document.drawCentredString(width / 2, height - 50, "Forecasting Report")

    # Metadata
    document.setFont("Helvetica", 12)
    y_position = height - 80
    document.drawString(50, y_position, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y_position -= 20
    document.drawString(50, y_position, f"Data File: {uploaded_file_name}")

    # Data Preview Section
    y_position -= 40
    document.setFont("Helvetica-Bold", 14)
    document.drawString(50, y_position, "Data Preview:")
    document.setFont("Helvetica", 12)
    y_position -= 20
    data_preview = data.head().to_string(index=False)
    dp_text = document.beginText(60, y_position)
    for line in data_preview.split('\n'):
        dp_text.textLine(line)
        y_position -= 15  # Adjust this value to increase line spacing
    document.drawText(dp_text)

    # Model Training Results
    y_position -= 40  # Increase space before starting this section
    document.setFont("Helvetica-Bold", 14)
    document.drawString(50, y_position, "Model Training Results:")
    document.setFont("Helvetica", 12)
    y_position -= 20
    for model_name, (model, rmse) in models.items():
        document.drawString(50, y_position, f"{model_name} RMSE: {rmse:.2f}")
        y_position -= 20  # Increase line spacing

    if isinstance(forecast, np.ndarray):
        forecast = forecast.item()

    # Best Model and Forecast
    y_position -= 30  # Increase space before this section
    document.setFont("Helvetica-Bold", 14)
    document.drawString(50, y_position, f"Best Performing Model: {best_model_name} with RMSE: {models[best_model_name][1]:.2f}")
    y_position -= 20
    document.drawString(50, y_position, f"Forecast for the next month: {forecast:.2f}")

    # Inserting Graph
    y_position -= 60  # Space before the graph
    document.setFont("Helvetica", 12)
    document.drawString(50, y_position, "Historical Data and Forecast:")
    y_position -= 20
    fig = go.Figure(data=[go.Scatter(x=data.index, y=data['y'], mode='lines', name='Actual')])
    fig.add_traces([go.Scatter(x=[data.index[-1], data.index[-1] + 1],
                               y=[data['y'].iloc[-1], forecast],
                               mode='lines+markers',
                               line=dict(color='red', dash='dash'),
                               name='Forecast')])
    fig.write_image(fig_path)
    img = Image(fig_path)
    img.drawHeight = 4 * inch  # Set image height
    img.drawWidth = 6 * inch   # Set image width
    img.wrapOn(document, width, height)
    img.drawOn(document, 50, y_position - 220)

    document.save()
    return filename

# Streamlit Application Code
st.set_page_config(page_title='Vendor Forecasting using Machine Learning', layout='wide' )

# To center an image using columns
col1, col2, col3 = st.columns([1,1,1])  # Adjust the proportions as needed
with col2:
    st.image(PILImage.open('se.png'), use_column_width=True)  # This will scale the image to fit the column width


st.title("Vendor Forecasting using Machine Learning")
uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.error("The uploaded file does not contain enough data to process.")
    else:
        models = {
            "ARIMA": train_arima(data['y']),
            "LSTM": train_lstm(data['y'], 5),
            "Linear Regression": train_linear_regression(data['y']),
            "Random Forest": train_random_forest(data['y'])
        }
        st.markdown("### Model Training Results")
        for model_name, (model, rmse) in models.items():
            st.markdown(f"**{model_name} RMSE:** {rmse:.2f}")

        best_model_name, best_model_info = min(models.items(), key=lambda x: x[1][1])
        st.markdown(f"### Best Performing Model: **{best_model_name}** with RMSE: **{best_model_info[1]:.2f}**")

        if best_model_name == "LSTM":
            n_steps = 5
            last_sequence = data['y'].values[-n_steps:]
            if len(last_sequence) < n_steps:
                st.error("Not enough data to make a forecast using LSTM.")
            else:
                X_new = np.array(last_sequence).reshape(1, n_steps, 1)
                forecast = models[best_model_name][0].predict(X_new)[0][0]
        else:
            last_index = data.index[-1]
            forecast = models[best_model_name][0].predict(np.array([[last_index + 1]]))[0]

        st.write(f"**Forecast for the next period:** {forecast:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['y'], mode='lines', name='Actual'))
        forecast_value = [data['y'].iloc[-1], forecast]
        fig.add_trace(go.Scatter(x=[data.index[-1], data.index[-1] + 1], y=forecast_value, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig)

        if st.button('Download Report'):
            report_file = create_pdf_report(data, models, best_model_name, forecast, 'forecast_plot.png', uploaded_file.name)
            with open(report_file, "rb") as file:
                st.download_button(
                    label="Download PDF Report",
                    data=file,
                    file_name="Time_Series_Forecast_Report.pdf",
                    mime="application/pdf"
                )

        openai.api_key = st.secrets["OPENAI_API_KEY"]
        st.header("Meet SCAI 🤖: Your Personal Supply Chain AI Assistant")
        query = st.text_input("Enter your supply chain-related query here:")
        if st.button("Submit"):
            if query:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Your name is SCAI. You are a highly specialized Supply Chain Assistant, expertly trained in all facets of supply chain management. Your expertise encompasses production planning, manufacturing processes, material resource planning (MRP), collaborative sales forecasting, vendor performance forecasting, and upstream supply chain planning. You possess a comprehensive understanding of logistics, inventory management, demand forecasting, and the integration of supply chain management with enterprise resource planning (ERP) systems. Additionally, Your role is to provide detailed, accurate, and practical advice on these topics. Please refrain from addressing questions outside this scope."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=512
                )
                answer = response['choices'][0]['message']['content']
                st.write(answer)
            else:
                st.write("Please enter a query to get an answer.")

# Footer
st.markdown("---")
st.markdown("Developed by Pranav Adiga and Adil Hussain")

# Adding a contact us button in the top-right corner
st.sidebar.header("Contact Us")
# Define the email link as a button with inline CSS styling
email_link = "pranav.adiga@non.se.com"  # Replace with your actual email
button_html = f"""<a href="mailto:{email_link}">
<button style='margin:0px; color: white; background-color: #FF6666; border: none; border-radius: 4px; padding: 10px 20px; cursor: pointer;'>
    Email
</button>
</a>"""
st.sidebar.markdown(button_html, unsafe_allow_html=True)
