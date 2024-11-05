import pandas as pd
import yfinance as yf
from statsmodels.datasets import co2
import requests
from datetime import datetime, timedelta

def get_air_quality():
    """Get daily air quality measurements from OpenAQ for a major city"""
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "location_id": "2178",  # London Westminster
        "parameter": "pm25",
        "date_from": (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        "limit": 1000
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame([{
        'date': item['date']['local'],
        'value': item['value']
    } for item in data['results']])
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()

def get_stock_data(symbol='AAPL', period='1y'):
    """Get daily stock prices using yfinance"""
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df[['Close']].rename(columns={'Close': 'price'})

def get_mauna_loa_co2():
    """Get the famous Mauna Loa CO2 measurements"""
    data = co2.load()
    return pd.Series(data.data, index=data.raw_data[:,0], name='co2_ppm')

def get_weather_data():
    """Get daily temperature data for New York City"""
    url = "https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')[['Mean_TemperatureF']].sort_index()

# Example usage showing multiple datasets
if __name__ == "__main__":
    # 1. Stock market data
    apple = get_stock_data('AAPL')
    print("\nApple Stock Prices (last 5 days):")
    print(apple.tail())
    
    # 2. CO2 Measurements
    co2_data = get_mauna_loa_co2()
    print("\nMauna Loa CO2 Measurements (last 5 observations):")
    print(co2_data.tail())
    
    # 3. Air Quality Data
    try:
        air_quality = get_air_quality()
        print("\nLondon Air Quality (last 5 measurements):")
        print(air_quality.tail())
    except:
        print("\nNote: Air quality API might require an API key or have usage limits")
    
    # 4. Weather Data
    weather = get_weather_data()
    print("\nSeattle Weather Data (last 5 days):")
    print(weather.tail())