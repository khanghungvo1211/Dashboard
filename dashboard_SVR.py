import os
import pandas as pd
import numpy as np
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import logging
import json
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Paths to data
folder_path = os.path.join('Data')
output_folder = os.path.join('Result')

# List of stock symbols
symbols = ['NVDA', 'INTC', 'PLTR', 'TSLA', 'AAPL', 'BBD', 'T', 'SOFI',
           'WBD', 'SNAP', 'NIO', 'BTG', 'F', 'AAL', 'NOK', 'BAC',
           'CCL', 'ORCL', 'AMD', 'PFE', 'KGC', 'MARA', 'SLB', 'NU',
           'MPW', 'MU', 'LCID', 'NCLH', 'RIG', 'AMZN', 'ABEV', 'U',
           'LUMN', 'AGNC', 'VZ', 'WBA', 'WFC', 'RIVN', 'UPST', 'GRAB',
           'CSCO', 'VALE', 'AVGO', 'PBR', 'GOOGL', 'SMMT', 'GOLD',
           'CMG', 'BCS', 'UAA']

# Load forecast_summary_svr.csv
forecast_summary_file = os.path.join(output_folder, 'forecast_summary_svr.csv')
if os.path.exists(forecast_summary_file):
    forecast_summary_df = pd.read_csv(forecast_summary_file)
    # Parse JSON-serialized lists into actual lists
    list_columns = ['Predicted_Prices', 'Test_Prices', 'Future_Price_Predictions', 'Train_Prices']
    for col in list_columns:
        if col in forecast_summary_df.columns:
            forecast_summary_df[col] = forecast_summary_df[col].apply(
                lambda x: json.loads(x) if pd.notnull(x) else []
            )
    logging.info("Loaded 'forecast_summary_svr.csv' successfully.")
else:
    forecast_summary_df = pd.DataFrame()
    logging.warning("Could not find 'forecast_summary_svr.csv'. Evaluation metrics and forecasts will not be displayed.")

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Define app layout
app.layout = html.Div([
    html.H1('GRU Stock Price Predictions Dashboard'),
    html.Div([
        html.Label('Select Stock Symbol'),
        dcc.Dropdown(
            id='stock-dropdown',
            options=[{'label': s, 'value': s} for s in symbols],
            value='NVDA'  # Default stock
        )
    ], style={'width': '25%', 'display': 'inline-block'}),
    html.Div([
        html.Label('Select Date Range'),
        dcc.DatePickerRange(
            id='date-picker',
            min_date_allowed='2014-09-18',  # Adjust based on your data
            max_date_allowed='2024-09-18',  # Adjust based on your data
            start_date='2014-09-18',        # Default start date
            end_date='2024-09-18'           # Default end date
        )
    ], style={'display': 'inline-block', 'marginLeft': '50px'}),
    html.Div([
        html.Label('Show 1-Year Forecast'),
        dcc.Checklist(
            id='forecast-checkbox',
            options=[{'label': 'Include 1-Year Forecast', 'value': 'show_forecast'}],
            value=[],  # Default unchecked
            inline=True
        )
    ], style={'marginTop': '20px'}),
    dcc.Graph(id='price-graph'),
    html.Div(id='metrics-output', style={'marginTop': '20px'})
])

# Define callback to update graph and evaluation metrics
@app.callback(
    [Output('price-graph', 'figure'),
     Output('metrics-output', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('forecast-checkbox', 'value')]
)
def update_graph(selected_stock, start_date, end_date, forecast_option):
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Get the forecast data for the selected stock
    forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]

    if forecast_row.empty:
        logging.error(f"No forecast data found for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"No forecast data found for {selected_stock}."

    # Parse 'Test_Prices', 'Predicted_Prices', 'Future_Price_Predictions', 'Train_Prices'
    test_prices = forecast_row.iloc[0]['Test_Prices']
    predicted_prices = forecast_row.iloc[0]['Predicted_Prices']
    future_prices = forecast_row.iloc[0]['Future_Price_Predictions']
    train_prices = forecast_row.iloc[0]['Train_Prices']

    # Load the stock data to get dates
    stock_data_file = os.path.join(folder_path, f'{selected_stock}.csv')
    if not os.path.exists(stock_data_file):
        logging.error(f"Stock data file not found for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"Stock data file not found for {selected_stock}."

    df_stock = pd.read_csv(stock_data_file)
    if 'Date' not in df_stock.columns or 'Close' not in df_stock.columns:
        logging.error(f"Incorrect data format in stock data file for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"Incorrect data format in stock data file for {selected_stock}."

    # Convert 'Date' to datetime
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    dates = df_stock['Date']
    data_length = len(df_stock)

    time_step = 60  # As used in your model
    total_samples = data_length - time_step

    # Calculate train_size and test_size based on 80:20 split
    train_size = int(total_samples * 0.8)
    test_size = total_samples - train_size

    # Ensure that the length of 'Test_Prices' matches 'test_size'
    if len(test_prices) != test_size or len(predicted_prices) != test_size:
        logging.error(f"Length mismatch between Test/Predict prices and test set for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"Length mismatch in data for {selected_stock}."

    # Get indices for train and test sets in original data
    train_indices_in_y_data = range(0, train_size)
    test_indices_in_y_data = range(train_size, train_size + test_size)
    indices_train = [i + time_step for i in train_indices_in_y_data]
    indices_test = [i + time_step for i in test_indices_in_y_data]

    # Ensure indices do not exceed the length of dates
    indices_train = [i for i in indices_train if i < len(dates)]
    indices_test = [i for i in indices_test if i < len(dates)]

    # Dates for train and test sets
    dates_train = dates.iloc[indices_train].reset_index(drop=True)
    dates_test = dates.iloc[indices_test].reset_index(drop=True)

    # Create DataFrame for train set
    df_train = pd.DataFrame({
        'Date': dates_train,
        'Train_Price': train_prices
    })

    # Create DataFrame for test set
    df_test = pd.DataFrame({
        'Date': dates_test,
        'Test_Price': test_prices,
        'Predicted_Price': predicted_prices
    })

    # Filter df_train and df_test based on 'start_date' and 'end_date'
    df_train_filtered = df_train[(df_train['Date'] >= start_date) & (df_train['Date'] <= end_date)]
    df_test_filtered = df_test[(df_test['Date'] >= start_date) & (df_test['Date'] <= end_date)]

    # Initialize data list for plotting
    data = []

    # Add Train Prices if available in the selected date range
    if not df_train_filtered.empty:
        data.append(
            go.Scatter(
                x=df_train_filtered['Date'],
                y=df_train_filtered['Train_Price'],
                mode='lines',
                name='Train Price',
                line=dict(color='red')
            )
        )

    # Add Actual Test Prices
    if not df_test_filtered.empty:
        data.append(
            go.Scatter(
                x=df_test_filtered['Date'],
                y=df_test_filtered['Test_Price'],
                mode='lines',
                name='Actual Test Price',
                line=dict(color='green')
            )
        )

    # Add Predicted Test Prices
    if not df_test_filtered.empty:
        data.append(
            go.Scatter(
                x=df_test_filtered['Date'],
                y=df_test_filtered['Predicted_Price'],
                mode='lines',
                name='Predicted Test Price',
                line=dict(color='blue')
            )
        )

    # Add future price predictions if selected
    if 'show_forecast' in forecast_option:
        # Generate future dates
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')

        if len(future_prices) != len(future_dates):
            logging.error(f"Length mismatch between future prices and future dates for {selected_stock}.")
        else:
            df_future = pd.DataFrame({
                'Date': future_dates,
                'Forecasted_Price': future_prices
            })

            # Filter future forecasts based on selected date range
            df_future_filtered = df_future[
                (df_future['Date'] >= start_date) & (df_future['Date'] <= end_date)
            ]

            if not df_future_filtered.empty:
                data.append(
                    go.Scatter(
                        x=df_future_filtered['Date'],
                        y=df_future_filtered['Forecasted_Price'],
                        mode='lines',
                        name='1-Year Forecast',
                        line=dict(dash='dash', color='orange')
                    )
                )
            else:
                logging.warning(f"No forecasted data within the selected date range for {selected_stock}.")

    # Create figure
    figure = {
        'data': data,
        'layout': go.Layout(
            title=f'{selected_stock} Price Prediction ({start_date.date()} to {end_date.date()})',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='closest'
        )
    }

    # Get RMSE, MSE, and MAPE
    try:
        rmse = float(forecast_row.iloc[0]['RMSE'])
        mse = float(forecast_row.iloc[0]['MSE'])
        mape = float(forecast_row.iloc[0]['MAPE'])
    except (ValueError, TypeError) as e:
        logging.error(f"Error parsing evaluation metrics for {selected_stock}: {e}")
        rmse = mse = mape = None

    # Prepare metrics output
    if rmse is not None and mse is not None and mape is not None:
        metrics_output = [
            html.P(f'Root Mean Squared Error (RMSE): {rmse:.4f}'),
            html.P(f'Mean Squared Error (MSE): {mse:.4f}'),
            html.P(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}')
        ]
    else:
        metrics_output = [html.P("Evaluation metrics are not available.")]

    return figure, metrics_output

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)

from dash import html, dcc
import plotly.graph_objs as go
import pandas as pd

def create_layout(app):
    # Load dữ liệu từ file CSV cho mô hình LSTM_SVR
    df = pd.read_csv('Result/forecast_summary_svr.csv')

    layout = html.Div([
        html.H2('LSTM_SVR Model Dashboard'),
        dcc.Graph(
            id='graph-lstm-svr',
            figure={
                'data': [
                    go.Scatter(
                        x=df['Date'],
                        y=df['Predicted'],
                        mode='lines+markers',
                        name='Predicted'
                    ),
                    go.Scatter(
                        x=df['Date'],
                        y=df['Actual'],
                        mode='lines',
                        name='Actual'
                    )
                ],
                'layout': go.Layout(
                    title='LSTM_SVR Model Predictions',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }
        )
    ])
    return layout