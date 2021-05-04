from tensorflow import keras
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from PIL import Image
import yfinance as yf
import plotly.graph_objs as go
import cufflinks as cf


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
**Description**

The Facebook stock data is obtained directly through Yahoo Finance API and they are updated live.

- App built by [Anson](https://www.linkedin.com/in/ansonnn07/)
- Built with `Python`, using `streamlit`, `yfinance`, `pandas`, `numpy`, `matplotlib`

**Links to my profiles**: [GitHub](https://github.com/ansonnn07/covid19-malaysia), 
[LinkedIn](https://www.linkedin.com/in/ansonnn07/),
[Kaggle](https://www.kaggle.com/ansonnn/code)
''')
st.markdown("""
**Tips about the figures**:
All the figures are ***interactive***. You can zoom in by dragging in the figure, and reset the axis by double-clicking. The legends can be clicked to disable or enable specific legends.
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
    df = pd.read_csv('fb_close.csv', index_col=[0], parse_dates=[0])
    return df


df = read_fb_stock()

full_df = yf.download(tickerSymbol)
st.header("**Facebook Stock Data History**")
full_styled = full_df.copy()
full_styled.index = full_styled.index.astype(str)
st.dataframe(full_styled)

fig = full_df['Close'].plot(title='Closing Price History of Facebook')
fig.update_layout(showlegend=False, xaxis_title=None,
                  yaxis_title=None, height=500)
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

st.header("**Ticker Data**")
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

st.header('**Bollinger Bands**')
qf = cf.QuantFig(today_df, title='First Quant Figure', legend='top', name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig, use_container_width=True)
