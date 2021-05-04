# Predicting Facebook Stock Price

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/ansonnn07/predicting-facebook-stock-price/main/app.py)

## Built with

<code><img height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="40" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="40" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="40" src="images//tensorflow-logo.png"></code>


<code><img height="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Plotly-logo-01-square.png/1200px-Plotly-logo-01-square.png"></code>
<code><img height="40" src="https://cdn.analyticsvidhya.com/wp-content/uploads/2020/10/image4.jpg"></code>

## Summary
This project is intend to apply time series forecasting algorithm by using Facebook Stock Price as an example. It is by no means a very practical method to help making decisions for investments. In fact, it is strictly not advisable to use this algorithm as a way to do so.

Models used involved: `ARIMA`, `SARIMAX`, `Prophet`, and `LSTM`. `LSTM` seems to perform very well when predicting ***one day*** into the future using **10 previous time steps**. It was able to capture the trends of rising and dropping in this case. Further explanation is available in the Web Application.

Thank you for checking out!

## Web Application
The Web App is accessible [here](https://share.streamlit.io/ansonnn07/predicting-facebook-stock-price/main/app.py) which you can directly see all the visualizations made.

## Some Visualizations
![New Cases](images//daily.png)
![New Cases](images//forecast.png)
