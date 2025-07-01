# mix_bag_app.py - Backend for Mix Bag Analytics

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime, timedelta
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- TEMPORARY CORS Configuration for Debugging ---
# WARNING: DO NOT USE IN PRODUCTION! This opens your API to all origins.
# This is a temporary step to definitively confirm if CORS is the blocking issue.
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow ALL origins
logging.warning("CORS configured to allow ALL origins (DEBUGGING MODE ONLY!). Remove this for production!")
print("DEBUG: CORS allowing ALL origins (*). REMOVE THIS FOR PRODUCTION!")
# --- END TEMPORARY CORS Configuration ---


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
        df['age_hours'] = df['age_timedelta'].dt.total_seconds() / 3600
        
        # Drop rows where bag_incoming_time_dt is NaT (invalid dates)
        df_cleaned = df.dropna(subset=['bag_incoming_time_dt']).copy()

        if df_cleaned.empty:
            return jsonify({"error": "No valid records found for analysis after processing 'bag_incoming_time'."}), 400

        # Create a string version of 'wbn' for safe unique counting
        df_cleaned['wbn_str'] = df_cleaned['wbn'].astype(str)

        # Group by bag_id to get aggregated info including is_bfsi status
        aggregation_dict = {
            'first_bag_incoming_time_dt': ('bag_incoming_time_dt', 'min'), # Earliest time for the bag
            'bag_age_hours': ('age_hours', 'min'), # Min age for the bag
            'bag_age_timedelta': ('age_timedelta', 'min'), # Min timedelta for the bag
            'wbn_count_per_bag': ('wbn_str', 'nunique') # Count unique WBNs per bag (total in bag)
        }
        if 'is_bfsi' in df_cleaned.columns:
            # If any WBN in the bag is BFSI, consider the bag as BFSI
            aggregation_dict['is_bfsi_bag'] = ('is_bfsi', 'max') 

        bag_age_info = df_cleaned.groupby('bag_id').agg(**aggregation_dict).reset_index()
        
        if 'bag_age_hours' not in bag_age_info.columns:
            bag_age_info['bag_age_hours'] = pd.Series(dtype='float64') # Ensure it exists if no data

        bag_age_info['wbn_count_per_bag'] = bag_age_info['wbn_count_per_bag'].fillna(0).astype(int)
        bag_age_info['is_bfsi_bag'] = bag_age_info['is_bfsi_bag'].fillna(False).astype(bool) # Ensure boolean type

        # Convert any remaining NaN in 'age_hours' to None before JSON serialization
        bag_age_info['age_hours'] = bag_age_info['bag_age_hours'].replace({np.nan: None})

        # Also get the cluster_id for each bag_id (assuming one cluster_id per bag_id)
        bag_cluster_map = df_cleaned[['bag_id', 'cluster_id']].drop_duplicates(subset=['bag_id']).copy()
        bag_age_info = pd.merge(bag_age_info, bag_cluster_map, on='bag_id', how='left')
        
        logging.info(f"Bag age info sample after aggregation: {bag_age_info.head().to_dict(orient='records')}")

        # --- Apply Filters to bag_age_info (before calculating overall totals and grouping) ---
        filtered_bag_age_info = bag_age_info.copy()

        # 1. WBN count filters
        if min_wbn_filter is not None:
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['wbn_count_per_bag'] >= min_wbn_filter]
        if max_wbn_filter is not None:
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['wbn_count_per_bag'] <= max_wbn_filter]

        # 2. BFSI filter
        if is_bfsi_filter is not None: # True or False explicitly passed
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['is_bfsi_bag'] == is_bfsi_filter]

        # 3. Cluster filter
        if selected_clusters and len(selected_clusters) > 0:
            # Handle 'UNKNOWN' cluster for filtering
            # Create a string representation for comparison that treats NaN as 'UNKNOWN'
            filtered_bag_age_info = filtered_bag_age_info[
                filtered_bag_age_info['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN').isin(selected_clusters)
            ]

        # 4. Ageing Bracket filter
        if selected_age_brackets and len(selected_age_brackets) > 0:
            age_filtered_bags = pd.DataFrame()
            for bracket in selected_age_brackets:
                if bracket == "0 to 1 hr":
                    age_filtered_bags = pd.concat([age_filtered_bags, filtered_bag_age_info[
                        (filtered_bag_age_info['bag_age_hours'] >= 0) & (filtered_bag_age_info['bag_age_hours'] <= 1)
                    ]])
                elif bracket == "1 to 2 hr":
                    age_filtered_bags = pd.concat([age_filtered_bags, filtered_bag_age_info[
                        (filtered_bag_age_info['bag_age_hours'] > 1) & (filtered_bag_age_info['bag_age_hours'] <= 2)
                    ]])
                elif bracket == "2 to 2.5 hrs":
                    age_filtered_bags = pd.concat([age_filtered_bags, filtered_bag_age_info[
                        (filtered_bag_age_info['bag_age_hours'] > 2) & (filtered_bag_age_info['bag_age_hours'] <= 2.5)
                    ]])
                elif bracket == "More than 2.5 hrs":
                    age_filtered_bags = pd.concat([age_filtered_bags, filtered_bag_age_info[
                        (filtered_bag_age_info['bag_age_hours'] > 2.5)
                    ]])
            # Remove duplicates if a bag falls into multiple selected brackets (unlikely for age but good practice)
            filtered_bag_age_info = age_filtered_bags.drop_duplicates(subset=['bag_id'])

        # Calculate Total Pending Bags and Shipments based on the *filtered* data
        total_pending_bags = len(filtered_bag_age_info)
        total_pending_shipments = int(filtered_bag_age_info['wbn_count_per_bag'].sum())


        # --- Ageing Analysis of Mix Bags (based on FILTERED data) ---
        ageing_groups = {
            "0 to 1 hr": {"bag_id_count": 0, "wbn_count": 0, "bags": []},
            "1 to 2 hr": {"bag_id_count": 0, "wbn_count": 0, "bags": []},
            "2 to 2.5 hrs": {"bag_id_count": 0, "wbn_count": 0, "bags": []},
            "More than 2.5 hrs": {"bag_id_count": 0, "wbn_count": 0, "bags": []}
        }

        def format_age_string(td: timedelta):
            """Formats a timedelta object into a human-readable age string."""
            if pd.isna(td): return "N/A"
            total_seconds = td.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            # Construct string based on available time units
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s ago"
            elif minutes > 0:
                return f"{minutes}m {seconds}s ago"
            else:
                return f"{seconds}s ago"


        for idx, row in filtered_bag_age_info.iterrows(): # Iterate over FILTERED data
            age = row['bag_age_hours']
            bag_id = row['bag_id']
            wbn_count_for_bag = row['wbn_count_per_bag']
            age_timedelta_obj = row['bag_age_timedelta']
            is_bfsi = row['is_bfsi_bag'] if 'is_bfsi_bag' in row else False
            cluster_id_for_bag = str(row['cluster_id']) if pd.notna(row['cluster_id']) else 'UNKNOWN'


            bag_data = {
                "bag_id": bag_id,
                "age_hours": age,
                "age_str": format_age_string(age_timedelta_obj),
                "wbn_count": wbn_count_for_bag, # This is the total WBNs in the bag
                "is_bfsi": is_bfsi,
                "cluster_id": cluster_id_for_bag # Include cluster_id for insights
            }

            # Categorize based on age
            if age is not None:
                if age >= 0 and age <= 1:
                    ageing_groups["0 to 1 hr"]["bag_id_count"] += 1
                    ageing_groups["0 to 1 hr"]["wbn_count"] += wbn_count_for_bag
                    ageing_groups["0 to 1 hr"]["bags"].append(bag_data)
                elif age > 1 and age <= 2:
                    ageing_groups["1 to 2 hr"]["bag_id_count"] += 1
                    ageing_groups["1 to 2 hr"]["wbn_count"] += wbn_count_for_bag
                    ageing_groups["1 to 2 hr"]["bags"].append(bag_data)
                elif age > 2 and age <= 2.5:
                    ageing_groups["2 to 2.5 hrs"]["bag_id_count"] += 1
                    ageing_groups["2 to 2.5 hrs"]["wbn_count"] += wbn_count_for_bag
                    ageing_groups["2 to 2.5 hrs"]["bags"].append(bag_data)
                elif age > 2.5:
                    ageing_groups["More than 2.5 hrs"]["bag_id_count"] += 1
                    ageing_groups["More than 2.5 hrs"]["wbn_count"] += wbn_count_for_bag
                    ageing_groups["More than 2.5 hrs"]["bags"].append(bag_data)
        
        # Sort bags within each ageing group by age (descending)
        for group_name in ageing_groups:
            valid_bags = [b for b in ageing_groups[group_name]["bags"] if b['age_hours'] is not None]
            ageing_groups[group_name]["bags"] = sorted(valid_bags, key=lambda x: x['age_hours'], reverse=True)


        # --- Cluster-wise Pending Counts and Details (based on FILTERED data) ---
        cluster_details = {}
        
        # Calculate WBNs per bag per cluster (from filtered_df_cleaned)
        # Re-derive this from the filtered set to ensure consistency
        filtered_bag_wbn_counts_for_cluster = df_cleaned[df_cleaned['bag_id'].isin(filtered_bag_age_info['bag_id'])].groupby(['bag_id', 'cluster_id'])['wbn_str'].nunique().reset_index(name='wbn_count_for_cluster')

        # Get unique bags per cluster and total WBNs per cluster from the filtered data
        # Use filtered_bag_age_info for bags, and filtered_bag_wbn_counts_for_cluster for cluster-specific WBNs
        
        # Get unique clusters from the filtered data
        unique_clusters_in_filtered_data = filtered_bag_age_info['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN').unique()

        for cluster_id_val_raw in unique_clusters_in_filtered_data:
            # Handle NaN cluster_id by treating it as 'UNKNOWN'
            cluster_display_id = str(cluster_id_val_raw) if pd.notna(cluster_id_val_raw) else 'UNKNOWN'

            # Get bags that are associated with this cluster in the FILTERED data
            bags_in_this_cluster_df = filtered_bag_age_info[
                filtered_bag_age_info['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN') == cluster_display_id
            ].copy()

            # Merge with wbn_counts_for_cluster to get cluster-specific WBNs for each bag
            merged_bags_data = pd.merge(
                bags_in_this_cluster_df,
                filtered_bag_wbn_counts_for_cluster[
                    filtered_bag_wbn_counts_for_cluster['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN') == cluster_display_id
                ],
                on='bag_id',
                how='left'
            )
            merged_bags_data['wbn_count_for_cluster'] = merged_bags_data['wbn_count_for_cluster'].fillna(0).astype(int)

            # Sort bags within this cluster by age_hours descending
            valid_bags_in_cluster_for_sort = [b for b in merged_bags_data.to_dict(orient='records') if b.get('bag_age_hours') is not None]
            sorted_bags_in_cluster = sorted(valid_bags_in_cluster_for_sort, key=lambda x: x['bag_age_hours'], reverse=True)
            
            formatted_bags = []
            for bag in sorted_bags_in_cluster:
                formatted_bags.append({
                    "bag_id": bag['bag_id'],
                    "wbn_count_per_bag": bag['wbn_count_per_bag'], # Total WBNs in the entire bag
                    "wbn_count_for_cluster": bag['wbn_count_for_cluster'], # WBNs specifically for this cluster within this bag
                    "age_str": format_age_string(bag['bag_age_timedelta']),
                    "age_hours": bag['bag_age_hours'],
                    "is_bfsi": bag['is_bfsi_bag'] if 'is_bfsi_bag' in bag else False,
                    "cluster_id": cluster_display_id # Ensure cluster_id is present
                })
            
            # Recalculate total bags and shipments for this specific cluster based on the FILTERED data
            current_cluster_total_bags = bags_in_this_cluster_df['bag_id'].nunique()
            current_cluster_total_shipments = merged_bags_data['wbn_count_for_cluster'].sum() # Sum of WBNs specifically for this cluster

            cluster_details[cluster_display_id] = {
                "bag_id_count": int(current_cluster_total_bags),
                "wbn_count": int(current_cluster_total_shipments),
                "bags": formatted_bags
            }
        
        # --- AI Insights Generation (based on FILTERED data) ---
        ai_insights = {
            "critical_clusters": {},
            "high_priority_bags": {}
        }

        # 1. Identify Critical Clusters
        # Focus on clusters with bags older than 2.5 hours and significant volume
        # Use filtered_bag_age_info which already contains bag_age_hours and wbn_count_per_bag
        if not filtered_bag_age_info.empty:
            
            # Calculate total WBNs, total bags, and average age per cluster
            cluster_summary_for_insights = filtered_bag_age_info.groupby('cluster_id').agg(
                total_bags=('bag_id', 'nunique'),
                total_wbns=('wbn_count_per_bag', 'sum'),
                avg_age_hours=('bag_age_hours', 'mean')
            ).reset_index()

            # Identify clusters where avg_age_hours > 2.5 and has at least 5 bags or 50 WBNs
            # These thresholds can be adjusted based on business needs
            critical_cluster_threshold_age = 2.5 
            critical_cluster_threshold_bags = 5
            critical_cluster_threshold_wbns = 50

            potential_critical_clusters = cluster_summary_for_insights[
                (cluster_summary_for_insights['avg_age_hours'] > critical_cluster_threshold_age) & 
                ((cluster_summary_for_insights['total_bags'] >= critical_cluster_threshold_bags) | 
                 (cluster_summary_for_insights['total_wbns'] >= critical_cluster_threshold_wbns))
            ].sort_values(by='avg_age_hours', ascending=[False]) # Sort by age to prioritize oldest

            for idx, row in potential_critical_clusters.iterrows():
                cluster_id = str(row['cluster_id']) if pd.notna(row['cluster_id']) else 'UNKNOWN'
                ai_insights["critical_clusters"][cluster_id] = {
                    "bags_count": int(row['total_bags']),
                    "wbn_count": int(row['total_wbns']),
                    "avg_age_hours": row['avg_age_hours']
                }

        # 2. Identify High Priority Outlier Bags (Age > 2.5 hrs and WBN Count >= 5)
        # Use filtered_bag_age_info directly
        outlier_age_threshold = 2.5 # bags older than 2.5 hours
        outlier_wbn_threshold = 5 # bags with 5 or more WBNs

        potential_outlier_bags = filtered_bag_age_info[
            (filtered_bag_age_info['bag_age_hours'] > outlier_age_threshold) &
            (filtered_bag_age_info['wbn_count_per_bag'] >= outlier_wbn_threshold)
        ].sort_values(by=['bag_age_hours', 'wbn_count_per_bag'], ascending=[False, False]) # Sort by age, then WBN count

        for idx, row in potential_outlier_bags.iterrows():
            bag_id = row['bag_id']
            ai_insights["high_priority_bags"][bag_id] = {
                "bag_id": bag_id,
                "age_hours": row['bag_age_hours'],
                "age_str": format_age_string(row['bag_age_timedelta']),
                "wbn_count": int(row['wbn_count_per_bag']),
                "is_bfsi": row['is_bfsi_bag'],
                "cluster_id": str(row['cluster_id']) if pd.notna(row['cluster_id']) else 'UNKNOWN'
            }
        
        # --- Update backend status variables on successful analysis ---
        TOTAL_ANALYSES_PERFORMED += 1
        LAST_ANALYSIS_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response_data = {
            "total_pending_bags": total_pending_bags,
            "total_pending_shipments": total_pending_shipments,
            "ageing_groups": ageing_groups,
            "cluster_details": cluster_details,
            "all_unique_clusters": [str(c) if pd.notna(c) else 'UNKNOWN' for c in df['cluster_id'].unique()], # Send all unique clusters from original data
            "ai_insights": ai_insights # NEW: Add AI insights to the response
        }

        return jsonify(response_data), 200

    except KeyError as e:
        logging.error(f"Missing expected column in CSV: {e}. Please ensure all required headers are present.", exc_info=True)
        return jsonify({"error": f"Missing expected column in CSV: {e}. Please ensure all required headers are present."}), 400
    except Exception as e:
        logging.error(f"An unhandled error occurred in mix_bag_analytics_api: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during mix bag analytics processing: {str(e)}"}), 500

# NEW: API endpoint for backend status
@app.route('/api/status', methods=['GET'])
def get_backend_status():
    uptime_seconds = (datetime.now() - APP_START_TIME).total_seconds()
    hours = int(uptime_seconds // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    uptime_str = f"{hours}h {minutes}m"

    return jsonify({
        "status": "online",
        "version": BACKEND_VERSION,
        "uptime": uptime_str,
        "last_analysis_time": LAST_ANALYSIS_TIME,
        "total_analyses": TOTAL_ANALYSES_PERFORMED
    }), 200


# This block ensures the Flask app runs locally when you execute this script directly.
# It will run on http://127.0.0.1:5000 (localhost:5000).
# In production (e.g., Render), gunicorn will call `app:app` directly, ignoring this `if __name__` block.
if __name__ == '__main__':
    # For local testing, you might temporarily set ALLOWED_ORIGIN here if needed, e.g.:
    # os.environ['ALLOWED_ORIGIN'] = 'http://127.0.0.1:5500' # If your frontend is on Live Server on port 5500
    app.run(debug=True, port=5000)
