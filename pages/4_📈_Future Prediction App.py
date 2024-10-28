import streamlit as st
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Livestock Methane Emission Future Prediction App", page_icon="🐮") 

# Title and description
st.title("Livestock Methane Emission Forecasting with Prophet Model")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("dataset/Cattle_CH4_dataset_cleaned_2000_2021.csv")

data = load_data()

# Function to create and display the Prophet model for a selected country
def create_prophet_model(country, df, periods):
    # Filter the data for the selected country
    country_data = df[df['Area'] == country]

    # Convert Year to datetime and then to an array, if needed
    country_data['Year'] = pd.to_datetime(country_data['Year'], format='%Y')

    # Check if the selected country has data
    if country_data.empty:
        st.warning(f"No data available for {country}. Please select another country.")
        return

    # Prepare the DataFrame for Prophet
    model_data = country_data[['Year', 'Value']].rename(columns={'Year': 'ds', 'Value': 'y'})

    # Initialize Prophet model
    model = Prophet()
    model.fit(model_data)

    # Create a dataframe for future predictions based on user input
    future = model.make_future_dataframe(periods=periods, freq='YE')  # Using 'YE' for year-end frequency
    forecast = model.predict(future)

    # Merge actual and predicted values for plotting
    forecast_plot_data = pd.merge(
        forecast[['ds', 'yhat']].rename(columns={'ds': 'Year', 'yhat': 'Forecast'}),
        model_data.rename(columns={'ds': 'Year', 'y': 'Original'}),
        on='Year',
        how='left'
    )

    # Calculate MAE and R² for historical data
    historical_data = forecast_plot_data.dropna()  # Exclude future data for metrics calculation
    mae = mean_absolute_error(historical_data['Original'], historical_data['Forecast'])
    r2 = r2_score(historical_data['Original'], historical_data['Forecast'])

    # Display metrics
    st.write(f"MAE for {country}: {mae}")
    st.write(f"R² for {country}: {r2}")

    # Plot using Plotly Express
    fig = px.line(forecast_plot_data, x='Year', y=['Original', 'Forecast'], 
                  labels={'value': 'Methane Emission Value (kt)', 'variable': 'Legend'},
                  title=f'{country} Methane Emission for the next {years_to_forecast} years')
    fig.update_layout(legend_title_text='Type')
    st.plotly_chart(fig)


# Country selection dropdown
country = st.selectbox("Select a Country", data['Area'].unique())

# Input for the number of years to forecast
years_to_forecast = st.number_input("Enter number of years to forecast", min_value=1, max_value=20, value=6, step=1)

# Run the model and display results
if st.button("Forecast"):
    create_prophet_model(country, data, periods=years_to_forecast)