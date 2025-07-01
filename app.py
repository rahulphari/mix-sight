# mix_bag_app.py - Backend for Mix Bag Analytics

from flask import Flask, request, jsonify
from flask_cors import CORS # Re-enabled Flask-CORS for proper, configurable CORS handling
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime, timedelta
# import os # Removed as it's not used in this version for environment variables

# Configure logging for better visibility into application behavior
logging.basicCon fig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# Configure CORS for online deployment.
# For production, it's recommended to specify allowed origins instead of '*'
# Example: CORS(app, origins=["https://yourfrontend.com", "http://localhost:3000"])
# For broad compatibility during initial deployment, '*' is used, but should be restricted.
CORS(app) 

# --- Global variables for status tracking ---
# These variables track the application's runtime status and analysis metrics.
APP_START_TIME = datetime.now()
BACKEND_VERSION = "1.5.0" # Updated version for enhanced filters and AI insights
TOTAL_ANALYSES_PERFORMED = 0
LAST_ANALYSIS_TIME = "Never"

@app.route('/api/mix-bag-analytics', methods=['POST'])
def mix_bag_analytics_api():
    """
    API endpoint for Mix Bag Analytics.
    Expects a JSON payload with 'file_name', 'csv_content', and new filter parameters.
    Processes the CSV data, applies filters, performs ageing and cluster analysis,
    and generates AI-driven insights.
    """
    global TOTAL_ANALYSES_PERFORMED, LAST_ANALYSIS_TIME # Declare global to modify these variables

    try:
        data = request.get_json()
        if not data:
            logging.error("Received empty or invalid JSON payload.")
            return jsonify({"error": "Invalid JSON payload"}), 400

        file_name = data.get('file_name')
        csv_content = data.get('csv_content')
        
        # New filter parameters extracted from the request payload
        min_wbn_filter = data.get('min_wbn_filter')
        max_wbn_filter = data.get('max_wbn_filter')
        is_bfsi_filter = data.get('is_bfsi_filter') # True, False, or None (for all)
        selected_clusters = data.get('selected_clusters') # List of cluster IDs
        selected_age_brackets = data.get('selected_age_brackets') # List of age bracket names

        if not file_name or not csv_content:
            logging.error("Missing file_name or csv_content in payload.")
            return jsonify({"error": "Missing file_name or csv_content in payload."}), 400

        logging.info(f"Received Mix Bag Analytics request: File='{file_name}', Filters: MinWBN={min_wbn_filter}, MaxWBN={max_wbn_filter}, BFSI={is_bfsi_filter}, Clusters={selected_clusters}, AgeBrackets={selected_age_brackets}")

        # --- CSV Reading and Initial Data Preprocessing ---
        try:
            # Read the CSV content into a pandas DataFrame
            df = pd.read_csv(io.StringIO(csv_content))
            # Standardize column names: lowercase, replace spaces with underscores, strip whitespace
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        except UnicodeDecodeError:
            logging.warning(f"Could not decode CSV file '{file_name}' as UTF-8. Trying with 'latin1'.")
            try:
                # Attempt to read again using 'latin1' encoding for broader compatibility
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

        # Define required columns for analysis
        required_columns = ['wbn', 'bag_id', 'bag_incoming_time', 'cluster_id']
        
        # If 'is_bfsi' column exists, ensure it's treated as boolean and add to required columns
        if 'is_bfsi' in df.columns:
            df['is_bfsi'] = df['is_bfsi'].astype(str).str.lower().isin(['true', '1', 'yes'])
            required_columns.append('is_bfsi')

        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {', '.join(missing_columns)} in CSV file.")
            return jsonify({"error": f"Missing required columns: {', '.join(missing_columns)}. Please check your CSV file headers."}), 400

        # --- Data Transformation and Age Calculation ---
        current_time = datetime.now()

        # Convert 'bag_incoming_time' to datetime objects, coercing errors to NaT (Not a Time)
        df['bag_incoming_time_dt'] = pd.to_datetime(df['bag_incoming_time'], errors='coerce')
        
        # Calculate age as a timedelta and then in hours
        df['age_timedelta'] = current_time - df['bag_incoming_time_dt']
        df['age_hours'] = df['age_timedelta'].dt.total_seconds() / 3600
        
        # Drop rows where 'bag_incoming_time_dt' is NaT (invalid dates) and create a copy to avoid SettingWithCopyWarning
        df_cleaned = df.dropna(subset=['bag_incoming_time_dt']).copy()

        if df_cleaned.empty:
            logging.warning("No valid records found for analysis after processing 'bag_incoming_time'.")
            return jsonify({"error": "No valid records found for analysis after processing 'bag_incoming_time'."}), 400

        # Create a string version of 'wbn' for accurate unique counting, especially if WBNs can be mixed types
        df_cleaned['wbn_str'] = df_cleaned['wbn'].astype(str)

        # Group by bag_id to get aggregated information for each bag
        aggregation_dict = {
            'first_bag_incoming_time_dt': ('bag_incoming_time_dt', 'min'), # Earliest incoming time for the bag
            'bag_age_hours': ('age_hours', 'min'), # Minimum age of any item within the bag (representing bag's age)
            'bag_age_timedelta': ('age_timedelta', 'min'), # Minimum timedelta for the bag
            'wbn_count_per_bag': ('wbn_str', 'nunique') # Count unique WBNs per bag
        }
        if 'is_bfsi' in df_cleaned.columns:
            # If any WBN in the bag is BFSI, consider the entire bag as BFSI (using max of boolean)
            aggregation_dict['is_bfsi_bag'] = ('is_bfsi', 'max') 

        bag_age_info = df_cleaned.groupby('bag_id').agg(**aggregation_dict).reset_index()
        
        # Ensure counts are integer and BFSI flag is boolean, handling potential NaNs
        bag_age_info['wbn_count_per_bag'] = bag_age_info['wbn_count_per_bag'].fillna(0).astype(int)
        bag_age_info['is_bfsi_bag'] = bag_age_info['is_bfsi_bag'].fillna(False).astype(bool) 

        # Merge cluster_id back into bag_age_info (assuming one cluster_id per bag_id)
        bag_cluster_map = df_cleaned[['bag_id', 'cluster_id']].drop_duplicates(subset=['bag_id']).copy()
        bag_age_info = pd.merge(bag_age_info, bag_cluster_map, on='bag_id', how='left')
        
        logging.info(f"Bag age info sample after aggregation: {bag_age_info.head().to_dict(orient='records')}")

        # --- Apply Filters to Aggregated Bag Data ---
        # All subsequent analysis will be based on this filtered dataset.
        filtered_bag_age_info = bag_age_info.copy()

        # 1. WBN count filters (min and max unique WBNs per bag)
        if min_wbn_filter is not None:
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['wbn_count_per_bag'] >= min_wbn_filter]
        if max_wbn_filter is not None:
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['wbn_count_per_bag'] <= max_wbn_filter]

        # 2. BFSI filter (filter based on whether the bag contains any BFSI shipments)
        if is_bfsi_filter is not None: # True or False explicitly passed
            filtered_bag_age_info = filtered_bag_age_info[filtered_bag_age_info['is_bfsi_bag'] == is_bfsi_filter]

        # 3. Cluster filter (filter by selected cluster IDs)
        if selected_clusters and len(selected_clusters) > 0:
            # Convert cluster_id to string, treating NaN as 'UNKNOWN' for consistent filtering
            filtered_bag_age_info = filtered_bag_age_info[
                filtered_bag_age_info['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN').isin(selected_clusters)
            ]

        # 4. Ageing Bracket filter (filter by selected age ranges)
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
            # Remove duplicates if a bag might fall into multiple selected brackets (e.g., if ranges overlap, though not in this case)
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
            """Formats a timedelta object into a human-readable age string (e.g., "1h 30m 15s ago")."""
            if pd.isna(td): return "N/A"
            total_seconds = td.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            
            # Construct string based on available time units for conciseness
            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s ago"
            elif minutes > 0:
                return f"{minutes}m {seconds}s ago"
            else:
                return f"{seconds}s ago"


        for idx, row in filtered_bag_age_info.iterrows(): # Iterate over the *FILTERED* data
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
                "wbn_count": wbn_count_for_bag, 
                "is_bfsi": is_bfsi,
                "cluster_id": cluster_id_for_bag 
            }

            # Categorize bags into defined ageing brackets
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
            
        # Sort bags within each ageing group by age (descending) for better readability
        for group_name in ageing_groups:
            valid_bags = [b for b in ageing_groups[group_name]["bags"] if b['age_hours'] is not None]
            ageing_groups[group_name]["bags"] = sorted(valid_bags, key=lambda x: x['age_hours'], reverse=True)


        # --- Cluster-wise Pending Counts and Details (based on FILTERED data) ---
        cluster_details = {}
        
        # Recalculate WBNs per bag per cluster from the original cleaned DataFrame,
        # but only for bags that are present in the filtered_bag_age_info.
        filtered_bag_wbn_counts_for_cluster = df_cleaned[df_cleaned['bag_id'].isin(filtered_bag_age_info['bag_id'])].groupby(['bag_id', 'cluster_id'])['wbn_str'].nunique().reset_index(name='wbn_count_for_cluster')

        # Get unique clusters from the *filtered* data to iterate through
        unique_clusters_in_filtered_data = filtered_bag_age_info['cluster_id'].apply(lambda x: str(x) if pd.notna(x) else 'UNKNOWN').unique()

        for cluster_id_val_raw in unique_clusters_in_filtered_data:
            # Handle NaN cluster_id by treating it as 'UNKNOWN' for display and comparison
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

        # 1. Identify Critical Clusters: Focus on clusters with bags older than a threshold
        # and significant volume (either bags count or WBN count).
        if not filtered_bag_age_info.empty:
            
            # Aggregate filtered data to get summary per cluster
            cluster_summary_for_insights = filtered_bag_age_info.groupby('cluster_id').agg(
                total_bags=('bag_id', 'nunique'),
                total_wbns=('wbn_count_per_bag', 'sum'),
                avg_age_hours=('bag_age_hours', 'mean')
            ).reset_index()

            # Define thresholds for identifying critical clusters
            critical_cluster_threshold_age = 2.5 # Average age of bags in cluster
            critical_cluster_threshold_bags = 5 # Minimum number of bags in cluster
            critical_cluster_threshold_wbns = 50 # Minimum number of WBNs in cluster

            potential_critical_clusters = cluster_summary_for_insights[
                (cluster_summary_for_insights['avg_age_hours'] > critical_cluster_threshold_age) & 
                ((cluster_summary_for_insights['total_bags'] >= critical_cluster_threshold_bags) | 
                 (cluster_summary_for_insights['total_wbns'] >= critical_cluster_threshold_wbns))
            ].sort_values(by='avg_age_hours', ascending=False) # Prioritize older clusters

            for idx, row in potential_critical_clusters.iterrows():
                cluster_id = str(row['cluster_id']) if pd.notna(row['cluster_id']) else 'UNKNOWN'
                ai_insights["critical_clusters"][cluster_id] = {
                    "bags_count": int(row['total_bags']),
                    "wbn_count": int(row['total_wbns']),
                    "avg_age_hours": row['avg_age_hours']
                }

        # 2. Identify High Priority Outlier Bags: Individual bags that are very old and contain many WBNs.
        outlier_age_threshold = 2.5 # Bags older than 2.5 hours
        outlier_wbn_threshold = 5 # Bags with 5 or more WBNs

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

        # Prepare the final response data
        response_data = {
            "total_pending_bags": total_pending_bags,
            "total_pending_shipments": total_pending_shipments,
            "ageing_groups": ageing_groups,
            "cluster_details": cluster_details,
            # Send all unique clusters from the original (unfiltered) data for filter options in frontend
            "all_unique_clusters": [str(c) if pd.notna(c) else 'UNKNOWN' for c in df['cluster_id'].unique()], 
            "ai_insights": ai_insights # Add AI insights to the response
        }

        return jsonify(response_data), 200

    except KeyError as e:
        logging.error(f"Missing expected column in CSV: {e}. Please ensure all required headers are present.", exc_info=True)
        return jsonify({"error": f"Missing expected column in CSV: {e}. Please ensure all required headers are present."}), 400
    except Exception as e:
        logging.error(f"An unhandled error occurred in mix_bag_analytics_api: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during mix bag analytics processing: {str(e)}"}), 500

# API endpoint for backend status monitoring
@app.route('/api/status', methods=['GET'])
def get_backend_status():
    """
    Provides current status of the backend application, including uptime, version,
    last analysis time, and total analyses performed.
    """
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
# For online deployment, you should use a production-ready WSGI server (e.g., Gunicorn, uWSGI)
# and set debug=False for security and performance.
if __name__ == '__main__':
    # For local development, you can keep debug=True.
    # For production, ensure debug=False.
    app.run(debug=False, port=5000)
