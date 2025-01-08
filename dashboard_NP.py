import os
import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import logging
from datetime import timedelta
import json

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

# Load forecast_summary_neural_prophet.csv
forecast_summary_file = os.path.join(output_folder, 'forecast_summary_neural_prophet.csv')
if os.path.exists(forecast_summary_file):
    forecast_summary_df = pd.read_csv(forecast_summary_file)
    # Parse JSON-serialized lists into actual lists using json.loads
    list_columns = ['Train_Prices', 'Test_Actual_Prices', 'Test_Predicted_Prices', 'Forecast_2024']
    for col in list_columns:
        if col in forecast_summary_df.columns:
            try:
                forecast_summary_df[col] = forecast_summary_df[col].apply(
                    lambda x: json.loads(x) if pd.notnull(x) else []
                )
                logging.info(f"Parsed column '{col}' successfully.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in column '{col}': {e}")
                forecast_summary_df[col] = [[] for _ in range(len(forecast_summary_df))]
    logging.info(f"Loaded '{forecast_summary_file}' successfully.")
else:
    forecast_summary_df = pd.DataFrame()
    logging.warning(f"Could not find '{forecast_summary_file}'. Evaluation metrics and forecasts will not be displayed.")

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Define app layout
app.layout = html.Div([
    html.H1('Stock Price Predictions Dashboard - XGBOOST'),
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
            min_date_allowed='2014-09-18',  # Adjusted min date for 10-year period
            max_date_allowed='2025-09-18',  # Extended by 1 year for forecasting
            start_date='2014-09-18',        # Default start date
            end_date='2025-09-18'           # Extended by 1 year for forecasting
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
    html.Div(id='metrics-output', style={'marginTop': '20px'}),
    html.Div([
        html.Button("Download Predictions", id="download-button"),
        dcc.Download(id="download-predictions")
    ], style={'marginTop': '20px'}),
    html.Div(id='download-info', style={'marginTop': '10px'})
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
    logging.info(f"Updating graph for stock: {selected_stock}")
    # Convert start_date and end_date to datetime
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        logging.info(f"Selected date range: {start_date} to {end_date}")
    except Exception as e:
        logging.error(f"Error parsing dates: {e}")
        return {}, "Error parsing dates."

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

    # Parse 'Test_Actual_Prices', 'Test_Predicted_Prices', 'Forecast_2024', 'Train_Prices'
    actual_prices = forecast_row.iloc[0].get('Test_Actual_Prices', [])
    predicted_prices = forecast_row.iloc[0].get('Test_Predicted_Prices', [])
    future_prices = forecast_row.iloc[0].get('Forecast_2024', [])
    train_prices = forecast_row.iloc[0].get('Train_Prices', [])

    logging.info(f"Lengths - Train: {len(train_prices)}, Test Actual: {len(actual_prices)}, Test Predicted: {len(predicted_prices)}, Forecast: {len(future_prices)}")

    # Check future_prices data
    if not future_prices:
        logging.warning(f"No future price predictions available for {selected_stock}.")
    else:
        logging.info(f"Loaded {len(future_prices)} future price predictions for {selected_stock}.")

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

    # Assuming the data was split into 80% train and 20% test
    train_size = int(len(dates) * 0.8)
    test_size = len(dates) - train_size

    logging.info(f"Total data length: {len(dates)}, Train size: {train_size}, Test size: {test_size}")

    # Ensure that the length of 'Test_Actual_Prices' matches 'test_size'
    if len(actual_prices) != test_size or len(predicted_prices) != test_size:
        logging.error(f"Length mismatch between actual/predicted prices and test set for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"Length mismatch in data for {selected_stock}."

    # Dates for train and test sets
    dates_train = dates.iloc[:train_size].reset_index(drop=True)
    dates_test = dates.iloc[train_size:].reset_index(drop=True)

    # Check if lengths match
    if len(train_prices) != len(dates_train):
        logging.warning(f"Length of train_prices ({len(train_prices)}) does not match dates_train ({len(dates_train)}). Adjusting.")
        min_len = min(len(train_prices), len(dates_train))
        train_prices = train_prices[:min_len]
        dates_train = dates_train[:min_len]

    if len(actual_prices) != len(dates_test) or len(predicted_prices) != len(dates_test):
        logging.warning(f"Length of test prices does not match dates_test for {selected_stock}. Adjusting.")
        min_len_test = min(len(actual_prices), len(predicted_prices), len(dates_test))
        actual_prices = actual_prices[:min_len_test]
        predicted_prices = predicted_prices[:min_len_test]
        dates_test = dates_test[:min_len_test]

    logging.info(f"After adjustment - Train: {len(train_prices)}, Test Actual: {len(actual_prices)}, Test Predicted: {len(predicted_prices)}")

    # Create DataFrame for train set
    df_train = pd.DataFrame({
        'Date': dates_train,
        'Train_Price': train_prices
    })

    # Create DataFrame for test set
    df_test = pd.DataFrame({
        'Date': dates_test,
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })

    # Filter df_train and df_test based on 'start_date' and 'end_date'
    df_train_filtered = df_train[(df_train['Date'] >= start_date) & (df_train['Date'] <= end_date)]
    df_test_filtered = df_test[(df_test['Date'] >= start_date) & (df_test['Date'] <= end_date)]

    logging.info(f"Filtered data - Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}")

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
        logging.info(f"Added Train Price trace with {len(df_train_filtered)} points.")

    # Add Actual Test Prices
    if not df_test_filtered.empty:
        data.append(
            go.Scatter(
                x=df_test_filtered['Date'],
                y=df_test_filtered['Actual_Price'],
                mode='lines',
                name='Actual Test Price',
                line=dict(color='green')
            )
        )
        logging.info(f"Added Actual Test Price trace with {len(df_test_filtered)} points.")

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
        logging.info(f"Added Predicted Test Price trace with {len(df_test_filtered)} points.")

    # Add future price predictions if selected
    if 'show_forecast' in forecast_option and future_prices:
        # Generate future dates
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')

        if len(future_prices) != len(future_dates):
            logging.error(f"Length mismatch between future prices ({len(future_prices)}) and future dates ({len(future_dates)}) for {selected_stock}.")
        else:
            df_future = pd.DataFrame({
                'Date': future_dates,
                'Future_Price': future_prices
            })
            # Filter future predictions based on date range
            df_future_filtered = df_future[(df_future['Date'] >= start_date) & (df_future['Date'] <= end_date)]
            logging.info(f"Future predictions after filtering: {len(df_future_filtered)}")
            if not df_future_filtered.empty:
                data.append(
                    go.Scatter(
                        x=df_future_filtered['Date'],
                        y=df_future_filtered['Future_Price'],
                        mode='lines',
                        name='1-Year Forecast',
                        line=dict(dash='dash', color='orange')
                    )
                )
                logging.info(f"Added 1-Year Forecast trace with {len(df_future_filtered)} points.")

    # Create figure
    figure = {
        'data': data,
        'layout': go.Layout(
            title=f'{selected_stock} Price Prediction - Neural Prophet ({start_date.date()} to {end_date.date()})',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='closest'
        )
    }

    # Get RMSE, MSE, and MAPE
    rmse = forecast_row.iloc[0].get('RMSE', None)
    mse = forecast_row.iloc[0].get('MSE', None)
    mape = forecast_row.iloc[0].get('MAPE', None)

    logging.info(f"Metrics - RMSE: {rmse}, MSE: {mse}, MAPE: {mape}")

    if rmse is not None and mse is not None and mape is not None:
        # Create metrics DataFrame
        metrics_df = pd.DataFrame({
            'Metric': ['RMSE', 'MSE', 'MAPE'],
            'Value': [f"{rmse:.4f}", f"{mse:.4f}", f"{mape:.2%}"]
        })

        metrics_output = html.Div([
            html.H4('Evaluation Metrics'),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in metrics_df.columns],
                data=metrics_df.to_dict('records'),
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_table={'width': '50%'}
            )
        ])
        logging.info("Added metrics table.")
    else:
        metrics_output = html.Div([
            html.H4('Evaluation Metrics'),
            html.P("Metrics not available.")
        ])
        logging.warning("Metrics not available.")

    return figure, metrics_output

# Define callback for downloading predictions
@app.callback(
    Output("download-predictions", "data"),
    [Input("download-button", "n_clicks")],
    [State('stock-dropdown', 'value')],
    prevent_initial_call=True,
)
def download_predictions(n_clicks, selected_stock):
    logging.info(f"Download button clicked for stock: {selected_stock}")
    if n_clicks is None:
        logging.info("No clicks detected.")
        return dash.no_update

    # Get the forecast data for the selected stock
    forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]

    if forecast_row.empty:
        logging.error(f"No forecast data found for {selected_stock}.")
        return dash.no_update

    # Extract data
    actual_prices = forecast_row.iloc[0].get('Test_Actual_Prices', [])
    predicted_prices = forecast_row.iloc[0].get('Test_Predicted_Prices', [])
    future_prices = forecast_row.iloc[0].get('Forecast_2024', [])
    train_prices = forecast_row.iloc[0].get('Train_Prices', [])

    logging.info(f"Lengths for download - Train: {len(train_prices)}, Test Actual: {len(actual_prices)}, Test Predicted: {len(predicted_prices)}, Forecast: {len(future_prices)}")

    # Load stock dates
    stock_data_file = os.path.join(folder_path, f'{selected_stock}.csv')
    if not os.path.exists(stock_data_file):
        logging.error(f"Stock data file not found for {selected_stock}.")
        return dash.no_update

    df_stock = pd.read_csv(stock_data_file)
    if 'Date' not in df_stock.columns or 'Close' not in df_stock.columns:
        logging.error(f"Incorrect data format in stock data file for {selected_stock}.")
        return dash.no_update

    # Convert 'Date' to datetime
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    dates = df_stock['Date']

    # Assuming the data was split into 80% train and 20% test
    train_size = int(len(dates) * 0.8)
    test_size = len(dates) - train_size

    # Ensure that the length of 'Test_Actual_Prices' matches 'test_size'
    if len(actual_prices) != test_size or len(predicted_prices) != test_size:
        logging.error(f"Length mismatch between actual/predicted prices and test set for {selected_stock}.")
        return dash.no_update

    # Prepare dates for test set
    dates_test = dates.iloc[train_size:].reset_index(drop=True)

    # Prepare DataFrame for download
    download_df = pd.DataFrame({
        'Date': dates_test,
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })

    logging.info(f"Prepared test set data for download with {len(download_df)} rows.")

    # Add future predictions if any
    if future_prices:
        last_date = dates.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')
        download_future_df = pd.DataFrame({
            'Date': future_dates,
            'Future_Price_Prediction': future_prices
        })
        download_df = pd.concat([download_df, download_future_df], ignore_index=True)
        logging.info(f"Added future predictions to download data with {len(download_future_df)} rows.")

    # Convert DataFrame to CSV
    logging.info(f"Sending download data for {selected_stock}.")
    return dcc.send_data_frame(download_df.to_csv, f'{selected_stock}_neural_prophet_predictions.csv', index=False)

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

from dash import html, dcc
import plotly.graph_objs as go
import pandas as pd

def create_layout(app):
    # Load dữ liệu từ file CSV cho mô hình Neural Prophet
    df = pd.read_csv('Result/forecast_summary_neural_prophet.csv')

    layout = html.Div([
        html.H2('Neural Prophet Model Dashboard'),
        dcc.Graph(
            id='graph-np',
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
                    title='Neural Prophet Model Predictions',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }
        )
    ])
    return layout