import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from datetime import datetime
import tiingo
from tiingo import TiingoClient
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

TIINGO_API_KEY = 'f88878d246bbcf9e6cba01ec8f27e9cfcfc79a34'

tiingo_config = {
    'session': True,
    'api_key': TIINGO_API_KEY,
}
client = TiingoClient(tiingo_config)

cur_date = datetime.now()
end_date = cur_date.strftime('%Y-%m-%d')

# end='2023-09-08'

start = '2010-01-01'

st.title('Stock Market Prediction')

user_input = st.text_input('Enter a Stock Symbol (e.g., NVDA. Use US Companies):')

if user_input:
    try:

        df = client.get_dataframe(user_input, startDate=start, endDate=end_date)

        st.subheader('Stock Data')
        st.write(df.tail())
        st.write(df.describe())

        st.subheader('Closing Price VS Time Chart')
        fig = plt.figure(figsize=(12, 8))
        plt.plot(df.close)
        plt.xlabel('Time')
        plt.ylabel('Price')
        #fig.layout.update(title_text)
        st.pyplot(fig)
        #st.area_chart(fig)

        st.subheader('Time Chart with 100 Value Mean Avg ')
        mone = df.close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 8))
        plt.plot(mone)
        plt.xlabel('Time')
        plt.ylabel('Price')
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100 Value Mean Avg ')
        mone = df.close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 8))
        plt.plot(mone)
        plt.plot(df.close)
        plt.xlabel('Time')
        plt.ylabel('Price')
        st.pyplot(fig)

        st.subheader('Closing Price vs Time Chart with 100 & 200 Value Mean Avg ')
        mone = df.close.rolling(100).mean()
        mtwo = df.close.rolling(200).mean()
        fig = plt.figure(figsize=(12, 8))
        plt.plot(mone, 'r', label='100')
        plt.plot(mtwo, 'g', label='200')
        plt.plot(df.close, 'b', label='Original')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Price')

        st.pyplot(fig)

        # 70:30  Split
        split_point = int(len(df) * 0.70)

        train = pd.DataFrame(df['close'][:split_point])
        test = pd.DataFrame(df['close'][split_point:])

        scaler = MinMaxScaler(feature_range=(0, 1))

        df_train_array = scaler.fit_transform(train)

        # Model Training
        x_train = []

        # Training To Be Predicted
        y_train = []

        for i in range(100, df_train_array.shape[0]):
            x_train.append(df_train_array[i - 100:i])
            y_train.append(df_train_array[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        seq_model = Sequential()

        seq_model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        seq_model.add(Dropout(0.2))

        # Long Short-Term Memory (LSTM) layer to the model.an RNN model

        # units -Specifies that the LSTM layer should have 50 hidden units (neurons).

        # ReLU function that will output the input directly if it is positive, otherwise, it will output zero.

        # Rerurn sequence - Tells that the LSTM layer should return sequences, which is typically used when stacking multiple LSTM layers.

        # Dropout technique used to prevent overfitting( Process where you do not feed more and more data which will overload the model).

        seq_model.add(LSTM(units=60, activation='relu', return_sequences=True))
        seq_model.add(Dropout(0.3))

        seq_model.add(LSTM(units=80, activation='relu', return_sequences=True, ))
        seq_model.add(Dropout(0.4))

        seq_model.add(LSTM(units=120, activation='relu'))
        seq_model.add(Dropout(0.5))

        seq_model.add(Dense(units=1))

        seq_model.compile(optimizer='adam', loss='mean_squared_error')
        seq_model.fit(x_train, y_train, epochs=3)

        past_100Days = train.tail(100)

        final_df = pd.concat([past_100Days, test], ignore_index=True)

        input_data = scaler.fit_transform(final_df)

        x_test = []

        y_test = []

        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = seq_model.predict(x_test)

        my_scaler = scaler.scale_
        scale_factor = 1 / my_scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        st.subheader('Predicted vs Original')
        fig2 = plt.figure(figsize=(12, 8))
        plt.plot(y_test, 'b', label='Original Price')
        plt.plot(y_predicted, 'r', label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)

        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)

        st.write(
            """
            <div style="position: absolute; bottom: 10px; right: 10px; text-align: right;">
                <p>Project Done by</p>
                <p>Jeevan Shetty [21BCAR0112]</p>
                <p>Arpitha Cherukuru [21BCAR0012]</p>
                <p>Siddharth Kannan [21BCAR0223]</p>
                <p>Chetan Vaishnav [21BCAR0017]</p>
            </div>
            """,
            unsafe_allow_html=True,
        )



    except tiingo.restclient.RestClientError as e:
        st.error(f"Tiingo API Error: {str(e)}")
