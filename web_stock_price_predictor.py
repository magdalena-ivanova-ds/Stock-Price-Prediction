import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Price Predictor App')

stock = st.text_input('Enter the stock ID', "GOOG")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download + normalization
def get_stock_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.set_index("Date")

    if "Adj Close" not in df.columns:
        _tmp = yf.download(
            ticker, start=start, end=end,
            auto_adjust=False, group_by="column",
            progress=False, threads=False
        )
        if isinstance(_tmp.columns, pd.MultiIndex):
            _tmp.columns = _tmp.columns.get_level_values(0)
        df = _tmp.copy()

    df = df.sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

google_data = get_stock_data(stock, start, end)

# Load model
model = load_model("Latest_stock_price_model.keras")

st.subheader("Stock Data")
st.write(google_data)

# Plots for moving averages
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, label=values.name if hasattr(values, "name") else None)
    plt.plot(full_data.Close, label="Close")
    if extra_data and extra_dataset is not None:
        plt.plot(extra_dataset, label=getattr(extra_dataset, "name", "extra"))
    if title:
        plt.title(title)
    plt.legend()
    return fig

st.subheader("Original Close Price and MA for 250 days")
google_data["MA_for_250_days"] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data["MA_for_250_days"], google_data, title="MA 250 vs Close"))

st.subheader("Original Close Price and MA for 200 days")
google_data["MA_for_200_days"] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data["MA_for_200_days"], google_data, title="MA 200 vs Close"))

st.subheader("Original Close Price and MA for 100 days")
google_data["MA_for_100_days"] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data["MA_for_100_days"], google_data, title="MA 100 vs Close"))

st.subheader("Original Close Price and MA for 100 days and MA for 250 days")
st.pyplot(plot_graph((15,6), google_data["MA_for_100_days"], google_data, 1, google_data["MA_for_250_days"], title="MAs vs Close"))

# Train/Test split
splitting_len = int(len(google_data) * 0.7)

train_close = google_data.Close.iloc[:splitting_len]
test_close  = google_data.Close.iloc[splitting_len:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_close.values.reshape(-1,1))

look_back = 100

x_train = []
y_train = []
for i in range(look_back, len(scaled_train)):
    x_train.append(scaled_train[i - look_back:i])
    y_train.append(scaled_train[i])
x_train, y_train = np.array(x_train), np.array(y_train)

# Build the test sequences
last_train_chunk = train_close.values[-look_back:]
concat_series = np.concatenate([last_train_chunk, test_close.values])
scaled_concat  = scaler.transform(concat_series.reshape(-1,1))

x_test = []
y_test = []
for i in range(look_back, len(scaled_concat)):
    x_test.append(scaled_concat[i - look_back:i])
    y_test.append(scaled_concat[i])
x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
train_pred_scaled = model.predict(x_train, verbose=0)
test_pred_scaled  = model.predict(x_test,  verbose=0)

# Invert scaling
train_pred = scaler.inverse_transform(train_pred_scaled).ravel()
test_pred  = scaler.inverse_transform(test_pred_scaled).ravel()

# Build aligned arrays for continuous plotting
N = len(google_data.Close)
train_predict_plot = np.full(N, np.nan, dtype=float)
test_predict_plot  = np.full(N, np.nan, dtype=float)

# Train predictions align to the train segment starting after 'look_back'
train_predict_plot[look_back:splitting_len] = train_pred

# Test predictions start at the split point (same index where test_close starts)
test_predict_plot[splitting_len:] = test_pred

# Tabular view of test original vs predicted
test_index = google_data.index[splitting_len:]
plot_df = pd.DataFrame(
    {
        "original_test_data": test_close.values[:len(test_pred)],
        "predictions": test_pred[:len(test_close.values)]
    },
    index=test_index[:len(test_pred)]
)

st.subheader("Original values vs Predicted values (Test)")
st.write(plot_df)

# Continuous plot of original vs predicted
st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15,6))
plt.plot(google_data.Close.values, label="Original Data")
plt.plot(train_predict_plot, label="Train Predict")
plt.plot(test_predict_plot, label="Test Predict")
plt.legend()
st.pyplot(fig)
