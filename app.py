# mix_bag_app.py - Backend for Mix Bag Analytics

from flask import Flask, request, jsonify
from flask_cors import CORS # Keep Flask-CORS imported, but we'll bypass its main config for a moment
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- EXTREMELY AGGRESSIVE CORS Configuration for Debugging ---
# WARNING: DO NOT USE IN PRODUCTION! This opens your API to all origins.
# This bypasses Flask-CORS's main configuration and forces the header.
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://rahulphari.github.io' # Explicitly set your frontend origin
    # You can also use '*' here if the above doesn't work, but try specific first
    # response.headers['Access-Control-Allow-Origin'] = '*' 
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true' # Important for some cases, though not directly used here
    print(f"DEBUG: Manually added CORS headers for origin: {response.headers['Access-Control-Allow-Origin']}")
    return response

# Handle OPTIONS requests (preflight) explicitly if not handled by Flask-CORS automatically
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    response = jsonify({'status': 'ok'})
    # The after_request decorator will add the CORS headers to this OPTIONS response too
    return response, 200

# You can comment out the previous CORS(app, ...) line entirely, or just leave it as is if it's currently `origins="*"`
# CORS(app, resources={r"/api/*": {"origins": "*"}}) # This line can be commented out or removed for this test

# --- END EXTREMELY AGGRESSIVE CORS Configuration ---


# --- Global variables for status tracking ---
APP_START_TIME = datetime.now()
BACKEND_VERSION = "1.5.0"
TOTAL_ANALYSES_PERFORMED = 0
LAST_ANALYSIS_TIME = "Never"

def sanitize_df_for_json(df_to_sanitize: pd.DataFrame):
    """
    Converts DataFrame columns to types that are JSON serializable,
    specifically handling NaT and NaN values.
    Returns a copy of the DataFrame.
    """
    sanitized_df = df_to_sanitize.copy()

    # Convert all numeric columns: replace np.nan with None
    for col in sanitized_df.select_dtypes(include=['number']).columns:
        sanitized_df[col] = sanitized_df[col].replace({np.nan: None})

    # Convert all datetime columns to string, handling NaT values
    for col in sanitized_df.select_dtypes(include=['datetime64[ns]']).columns:
        sanitized_df[col] = sanitized_df[col].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else None)

    # Convert any remaining None/NaN-like strings in object/string columns to actual None
    for col in sanitized_df.select_dtypes(include=['object']).columns:
        sanitized_df[col] = sanitized_df[col].astype(str).replace({'nan': None, '': None, 'None': None}, regex=False)
        sanitized_df[col] = sanitized_df[col].replace({pd.NA: None})
    
    return sanitized_df

@app.route('/api/mix-bag-analytics', methods=['POST'])
def mix_bag_analytics_api():
    """
    API endpoint for Mix Bag Analytics.
    Expects a JSON payload with 'file_name', 'csv_content', and new filter parameters.
    """
    global TOTAL_ANALYSES_PERFORMED, LAST_ANALYSIS_TIME # Declare global to modify

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        file_name = data.get('file_name')
        csv_content = data.get('csv_content')
        
        # New filter parameters
        min_wbn_filter = data.get('min_wbn_filter')
        max_wbn_filter = data.get('max_wbn_filter')
        is_bfsi_filter = data.get('is_bfsi_filter') # True, False, or None (for all)
        selected_clusters = data.get('selected_clusters') # List of cluster IDs
        selected_age_brackets = data.get('selected_age_brackets') # List of age bracket names

        if not file_name or not csv_content:
            return jsonify({"error": "Missing file_name or csv_content in payload."}), 400

        logging.info(f"Received Mix Bag Analytics request: File='{file_name}', Filters: MinWBN={min_wbn_filter}, MaxWBN={max_wbn_filter}, BFSI={is_bfsi_filter}, Clusters={selected_clusters}, AgeBrackets={selected_age_brackets}")

        # Read the CSV content into a pandas DataFrame
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            # Standardize column names (lowercase, replace spaces with underscores)
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        except UnicodeDecodeError:
            logging.error(f"Could not decode CSV file '{file_name}' as UTF-8. Trying with 'latin1'.")
            try:
                # Read again using latin1 encoding
                df = pd.read_csv(io.StringIO(csv_content.decode('latin1')))
                df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
            except Exception as decode_err:
                logging.error(f"Could not decode CSV file '{file_name}' with latin1 either: {decode_err}")
                return jsonify({"error": f"Failed to decode CSV file: {file_name}. Ensure it's a valid text-based CSV (UTF-8 or Latin-1)."}), 400
        except pd.errors.EmptyDataError:
            logging.warning(f"Uploaded CSV file '{file_name}' is empty.")
            return jsonify({"error": "The uploaded CSV file is empty or contains no data."}), 400
        except pd.errors.ParserError as e:
            logging.error(f"Parsing error for uploaded CSV file '{file_name}': {e}")
            return jsonify({"error": f"Error parsing uploaded CSV file: {file_name}. Please check file format and headers."}), 400

        # Add 'is_bfsi' to required_columns if it exists in the CSV
        required_columns = ['wbn', 'bag_id', 'bag_incoming_time', 'cluster_id']
        if 'is_bfsi' in df.columns:
            # Ensure 'is_bfsi' is boolean
            df['is_bfsi'] = df['is_bfsi'].astype(str).str.lower().isin(['true', '1', 'yes'])
            required_columns.append('is_bfsi')


        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing required columns: {', '.join(missing_columns)}. Please check your CSV file headers."}), 400

        # --- Data Preprocessing ---
        current_time = datetime.now()

        # Convert 'bag_incoming_time' to datetime, coerce errors to NaT
        df['bag_incoming_time_dt'] = pd.to_datetime(df['bag_incoming_time'], errors='coerce')
        
        # Calculate age in hours, handling NaT values
        df['age_timedelta'] = current_time - df['bag_incoming_time_dt']
        df
