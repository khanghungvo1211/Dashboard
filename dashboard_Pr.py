import os
import pandas as pd
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import logging
from datetime import timedelta
import ast

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

# Load forecast_summary_Prophet.csv
forecast_summary_file = os.path.join(output_folder, 'forecast_summary_Prophet.csv')
if os.path.exists(forecast_summary_file):
    forecast_summary_df = pd.read_csv(forecast_summary_file)
    # Parse string lists into actual lists using ast.literal_eval
    list_columns = ['Predicted_Prices', 'Actual_Prices', 'Future_Price_Predictions', 'Train_Prices']
    for col in list_columns:
        if col in forecast_summary_df.columns:
            try:
                forecast_summary_df[col] = forecast_summary_df[col].apply(
                    lambda x: ast.literal_eval(x) if pd.notnull(x) else []
                )
                logging.info(f"Parsed column '{col}' successfully.")
            except (ValueError, SyntaxError) as e:
                logging.error(f"Error parsing column '{col}': {e}")
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
    html.H1('Stock Price Predictions Dashboard (Prophet)'),
    
    # Controls
    html.Div([
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
                max_date_allowed='2025-12-31',  # Extended max date to accommodate future forecasts
                start_date='2014-09-18',        # Default start date
                end_date='2025-09-18'           # Extended default end date to include 1-year forecast
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
        ], style={'marginTop': '20px'})
    ]),
    
    # Graph
    dcc.Graph(id='price-graph'),
    
    # Evaluation Metrics
    html.Div(id='metrics-output', style={'marginTop': '20px'}),
    
    # Download Button
    html.Div([
        html.Button("Download Predictions", id="download-button"),
        dcc.Download(id="download-predictions")
    ], style={'marginTop': '20px'}),
    
    # Download Info
    html.Div(id='download-info', style={'marginTop': '10px'})
])

# Callback to update graph and metrics
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
    
    # Convert dates to datetime
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        logging.info(f"Selected date range: {start_date} to {end_date}")
    except Exception as e:
        logging.error(f"Error parsing dates: {e}")
        return {}, "Error parsing dates."
    
    # Retrieve forecast data
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
    
    # Extract and parse forecast data
    actual_prices = forecast_row.iloc[0]['Actual_Prices']
    predicted_prices = forecast_row.iloc[0]['Predicted_Prices']
    future_prices = forecast_row.iloc[0]['Future_Price_Predictions']
    train_prices = forecast_row.iloc[0]['Train_Prices']
    
    logging.info(f"Lengths - Train: {len(train_prices)}, Actual: {len(actual_prices)}, Predicted: {len(predicted_prices)}, Future: {len(future_prices)}")
    
    # Load stock data
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
    
    # Process stock data
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock_sorted = df_stock.sort_values('Date')
    dates = df_stock_sorted['Date']
    
    # Determine train-test split date based on 8-year training period
    split_date = df_stock_sorted['Date'].min() + pd.DateOffset(years=8)
    logging.info(f"Train-Test split date: {split_date}")
    
    # Split the data
    train_mask = df_stock_sorted['Date'] <= split_date
    test_mask = df_stock_sorted['Date'] > split_date
    
    df_train = df_stock_sorted[train_mask]
    df_test = df_stock_sorted[test_mask]
    
    # Align forecast data with test set dates
    test_dates = df_test['Date'].reset_index(drop=True)
    actual_prices = actual_prices[:len(test_dates)]
    predicted_prices = predicted_prices[:len(test_dates)]
    
    if len(actual_prices) != len(predicted_prices):
        logging.error(f"Length mismatch between actual and predicted prices for {selected_stock}.")
        return {
            'data': [],
            'layout': go.Layout(
                title=f'{selected_stock} Price Prediction',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Price'}
            )
        }, f"Length mismatch in data for {selected_stock}."
    
    # Filter train and test data based on selected date range
    df_train_filtered = df_train[(df_train['Date'] >= start_date) & (df_train['Date'] <= end_date)]
    df_test_filtered = pd.DataFrame({
        'Date': test_dates[:len(actual_prices)],
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })
    df_test_filtered = df_test_filtered[
        (df_test_filtered['Date'] >= start_date) & (df_test_filtered['Date'] <= end_date)
    ]
    
    logging.info(f"Filtered data - Train: {len(df_train_filtered)}, Test: {len(df_test_filtered)}")
    
    # Initialize plot data
    plot_data = []
    
    # Add Train Prices
    if not df_train_filtered.empty:
        plot_data.append(
            go.Scatter(
                x=df_train_filtered['Date'],
                y=df_train_filtered['Close'],
                mode='lines',
                name='Train Price',
                line=dict(color='red')
            )
        )
        logging.info(f"Added Train Price trace with {len(df_train_filtered)} points.")
    
    # Add Actual Test Prices
    if not df_test_filtered.empty:
        plot_data.append(
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
        plot_data.append(
            go.Scatter(
                x=df_test_filtered['Date'],
                y=df_test_filtered['Predicted_Price'],
                mode='lines',
                name='Predicted Test Price',
                line=dict(color='blue')
            )
        )
        logging.info(f"Added Predicted Test Price trace with {len(df_test_filtered)} points.")
    
    # Add vertical split line
    if pd.notnull(split_date):
        plot_data.append(
            go.Scatter(
                x=[split_date, split_date],
                y=[df_stock_sorted['Close'].min(), df_stock_sorted['Close'].max()],
                mode='lines',
                name='Train-Test Split',
                line=dict(color='black', dash='dash')
            )
        )
        logging.info("Added Train-Test Split line.")
    
    # Add 1-Year Forecast if selected
    if 'show_forecast' in forecast_option and future_prices:
        # Generate future business days (assuming 252 trading days in a year)
        last_date = df_stock_sorted['Date'].max()
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices))
        
        if len(future_prices) != len(future_dates):
            logging.error(f"Length mismatch between future prices ({len(future_prices)}) and future dates ({len(future_dates)}) for {selected_stock}.")
        else:
            df_future = pd.DataFrame({
                'Date': future_dates,
                'Forecasted_Price': future_prices
            })
            # Filter future forecasts based on date range
            df_future_filtered = df_future[
                (df_future['Date'] >= start_date) & (df_future['Date'] <= end_date)
            ]
            if not df_future_filtered.empty:
                plot_data.append(
                    go.Scatter(
                        x=df_future_filtered['Date'],
                        y=df_future_filtered['Forecasted_Price'],
                        mode='lines',
                        name='1-Year Forecast',
                        line=dict(dash='dash', color='orange')
                    )
                )
                logging.info(f"Added 1-Year Forecast trace with {len(df_future_filtered)} points.")
            else:
                logging.warning(f"No forecasted data within the selected date range for {selected_stock}.")
    
    # Create figure
    figure = {
        'data': plot_data,
        'layout': go.Layout(
            title=f'{selected_stock} Price Prediction ({start_date.date()} to {end_date.date()})',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='closest'
        )
    }
    
    # Extract evaluation metrics
    rmse = forecast_row.iloc[0].get('RMSE', None)
    mse = forecast_row.iloc[0].get('MSE', None)
    mape = forecast_row.iloc[0].get('MAPE', None)
    
    logging.info(f"Metrics - RMSE: {rmse}, MSE: {mse}, MAPE: {mape}")
    
    # Prepare metrics output
    if rmse is not None and mse is not None and mape is not None:
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

# Callback to handle download
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

    # Retrieve forecast data
    forecast_row = forecast_summary_df[forecast_summary_df['Symbol'] == selected_stock]
    
    if forecast_row.empty:
        logging.error(f"No forecast data found for {selected_stock}.")
        return dash.no_update
    
    # Extract data
    actual_prices = forecast_row.iloc[0].get('Actual_Prices', [])
    predicted_prices = forecast_row.iloc[0].get('Predicted_Prices', [])
    future_prices = forecast_row.iloc[0].get('Future_Price_Predictions', [])
    train_prices = forecast_row.iloc[0].get('Train_Prices', [])
    
    logging.info(f"Lengths for download - Train: {len(train_prices)}, Actual: {len(actual_prices)}, Predicted: {len(predicted_prices)}, Future: {len(future_prices)}")
    
    # Load stock data
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
    
    # Determine train-test split date based on 8-year training period
    split_date = df_stock['Date'].min() + pd.DateOffset(years=8)
    logging.info(f"Train-Test split date: {split_date}")
    
    # Split the data
    train_mask = df_stock['Date'] <= split_date
    test_mask = df_stock['Date'] > split_date
    
    df_train = df_stock[train_mask]
    df_test = df_stock[test_mask]
    
    # Align forecast data with test set dates
    test_dates = df_test['Date'].reset_index(drop=True)
    actual_prices = actual_prices[:len(test_dates)]
    predicted_prices = predicted_prices[:len(test_dates)]
    
    if len(actual_prices) != len(predicted_prices):
        logging.error(f"Length mismatch between actual and predicted prices for {selected_stock}.")
        return dash.no_update
    
    # Prepare DataFrame for download
    df_test_download = pd.DataFrame({
        'Date': test_dates[:len(actual_prices)],
        'Actual_Price': actual_prices,
        'Predicted_Price': predicted_prices
    })
    
    logging.info(f"Prepared test set data for download with {len(df_test_download)} rows.")
    
    # Add future predictions if any
    if future_prices:
        last_date = df_stock['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_prices), freq='D')
        download_future_df = pd.DataFrame({
            'Date': future_dates,
            'Future_Price_Prediction': future_prices
        })
        df_download = pd.concat([df_test_download, download_future_df], ignore_index=True)
        logging.info(f"Added future predictions to download data with {len(download_future_df)} rows.")
    else:
        df_download = df_test_download
        logging.info("No future predictions to add to download data.")
    
    # Convert DataFrame to CSV and send for download
    logging.info(f"Sending download data for {selected_stock}.")
    return dcc.send_data_frame(df_download.to_csv, f'{selected_stock}_Prophet_predictions.csv', index=False)

# Run Dash app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

from dash import html, dcc
import plotly.graph_objs as go
import pandas as pd

def create_layout(app):
    # Load dữ liệu từ file CSV cho mô hình Prophet
    df = pd.read_csv('Result/forecast_summary_Prophet.csv')

    layout = html.Div([
        html.H2('Prophet Model Dashboard'),
        dcc.Graph(
            id='graph-pr',
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
                    title='Prophet Model Predictions',
                    xaxis={'title': 'Date'},
                    yaxis={'title': 'Price'}
                )
            }
        )
    ])
    return layout