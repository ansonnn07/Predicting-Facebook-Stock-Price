from tensorflow import keras
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import yfinance as yf
import plotly.graph_objs as go
import cufflinks as cf
import pickle


register_matplotlib_converters()
# sns.set(style='whitegrid', palette='muted', font_scale=1.5)
# plt.rcParams['figure.figsize'] = 15, 10

pd.options.plotting.backend = "plotly"

st.set_page_config(
    page_title="Facebook Stock",
    page_icon="FB",
    layout="wide",
    initial_sidebar_state="expanded",
)


np.random.seed(42)
tf.random.set_seed(42)

# Add title, descriptions and image
st.title('COVID-19 Malaysia')
st.markdown('''
- The **Facebook stock data** is obtained directly through Yahoo Finance API and they provide live updates.
- This app is built to show the application of some **time series forecasting algorithms**.
- **DISCLAIMER**: This is just an attempt to forecast a time series model, 
it is strictly not advisable to rely solely on this to make decisions on investing assets. 
- The **live updates** to the Facebook stock data can also be seen here.
- A figure with **Bollinger Bands** is also provided, as Bollinger Bands is one of the methods used to determine whether
it is beneficial to buy or sell at a given time.


- App built by [Anson](https://www.linkedin.com/in/ansonnn07/)
- Built with `Python`, using `streamlit`, `yfinance`, `pandas`, `numpy`, `plotly`

**Links**: [GitHub](https://github.com/ansonnn07/Predicting-Facebook-Stock-Price), 
[LinkedIn](https://www.linkedin.com/in/ansonnn07/),
[Kaggle](https://www.kaggle.com/ansonnn/code)
''')
st.markdown("""
**Tips about the figures**:
All the figures are ***interactive***. You can **zoom in** by dragging in the figure,
and **reset** the axis by double-clicking. 
The **legends** can also be clicked to disable or enable specific legends.
""")

st.markdown('---')

tickerSymbol = "FB"
tickerData = yf.Ticker(tickerSymbol)

imageURL = tickerData.info['logo_url']
logo_html = f'<img src={imageURL}><br>'
st.markdown(logo_html, unsafe_allow_html=True)

companyName = tickerData.info['longName']
st.header(f"**{companyName}**")

companyInfo = tickerData.info['longBusinessSummary']
st.info(companyInfo)


@st.cache
def read_fb_stock():
    return pd.read_csv('data//fb_close.csv', index_col=[0], parse_dates=[0])


def load_model():
    return keras.models.load_model('data//LSTM_model.h5')


@st.cache
def load_scaler():
    return pickle.load(open('data//scaler.pkl', 'rb'))


@st.cache
def load_results():
    return pd.read_csv('data//full_results.csv', index_col=[0], parse_dates=[0])


df = read_fb_stock()
model = load_model()
scaler = load_scaler()
results_df = load_results()

full_df = yf.download(tickerSymbol)
st.header("**Facebook Stock Data History**")
full_styled = full_df.copy()
full_styled.index = full_styled.index.astype(str)
full_styled = full_styled.style.format('{0:,.2f}')
st.dataframe(full_styled)


fig = full_df['Close'].plot(title='Closing Price History of Facebook')
fig.update_layout(showlegend=False, xaxis_title=None,
                  yaxis_title=None, height=600)
fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(fig, use_container_width=True)

df_roll = df['Close'].rolling(7, min_periods=1, center=True).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close",
                         mode='markers', marker_size=3))
fig.add_trace(go.Scatter(x=df_roll.index, y=df_roll.values, name="Average",
                         line_width=1.6))
fig.update_layout(title='Daily Closing Price VS 7-day Moving Average',
                  hovermode='x',
                  height=600,
                  legend=dict(
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.01
                  ))
st.plotly_chart(fig, use_container_width=True)

st.header("**Forecasting Results**")
results_df.rename(columns={'Close': 'Closing Price',
                           'LSTM_pred': 'LSTM Predictions',
                           'LSTM_future': 'LSTM Future Forecast'}, inplace=True)
fig = results_df.iloc[-365:].plot(title='Predicting Facebook Stock Closing Price\n'
                                  'using Different Models')
fig.update_layout(height=600,
                  xaxis_title=None, yaxis_title=None,
                  hovermode='x',
                  legend=dict(title=None,
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01
                              ))
fig.update_traces(line_width=1.6, hovertemplate=None)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Explanation for the predictive models**

- **ARIMA** and **SARIMAX** (red and green lines) are common machine learning models used for time series forecasting.
In this case, they did not perform so well as compared to the other models.
- **Prophet** is another type of model developed by Facebook themselves, that also did not perform as well,
because Prophet is designed specifically for time series that has very obvious **seasonality** 
characteristics and **consistent overall average**,
unlike the current time series, where there is a clear **trend of increasing** upwards.
- **LSTM** (Long Short Term Memory) is a deep learning model, more specifically a type of Recurrent Neural Networks.
it shows promising results predicting the overall values of the stock price, 
with a **root mean square error** of **9.2134** on the **predictions** when given the ***actual*** data.
This is because the LSTM in this case uses **previous 10 days** to predict ***one day*** into the future, 
therefore, the results will be very close to the original closing price 
as it can capture the rising and dropping trend.
When **forecasting** the stock price of the **future 30 days** (blue lines) using only the **actual data of 10 days**, 
it does show acceptable results but it has to be validated later when the actual data has been obtained.
""")

st.header("**Data Today**")
today_df = yf.download(tickerSymbol, period='1d', interval='1m')
fig = go.Figure(data=[go.Candlestick(x=today_df.index,
                open=today_df['Open'],
                high=today_df['High'],
                low=today_df['Low'],
                close=today_df['Close'])])
fig.update_layout(title="Facebook's Live Stock Price Today",
                  xaxis_title=None, yaxis_title=None,
                  height=600,
                  legend=dict(
                      yanchor="top",
                      y=0.99,
                      xanchor="left",
                      x=0.01
                  ))
fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=30, label="30m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(fig, use_container_width=True)

st.subheader('**Bollinger Bands**')
qf = cf.QuantFig(today_df, legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig, use_container_width=True)
