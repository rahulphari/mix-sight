# mix_bag_app.py - Backend for Mix Bag Analytics

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for development

# --- Global variables for status tracking ---
APP_START_TIME = datetime.now()
BACKEND_VERSION = "1.8.0" # Version with Ageing Group fix
TOTAL_ANALYSES_PERFORMED = 0
LAST_ANALYSIS_TIME = "Never"


@app.route('/api/mix-bag-analytics', methods=['POST'])
def mix_bag_analytics_api():
    """
    API endpoint for Mix Bag Analytics.
    Corrects the WBN counting logic for clusters and restores ageing group calculations.
    """
    global TOTAL_ANALYSES_PERFORMED, LAST_ANALYSIS_TIME

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        file_name = data.get('file_name')
        csv_content = data.get('csv_content')
        
        min_wbn_filter = data.get('min_wbn_filter')
        max_wbn_filter = data.get('max_wbn_filter')
        is_bfsi_filter = data.get('is_bfsi_filter')
        selected_clusters = data.get('selected_clusters')
        selected_age_brackets = data.get('selected_age_brackets')

        if not file_name or not csv_content:
            return jsonify({"error": "Missing file_name or csv_content in payload."}), 400

        # Read and preprocess the main DataFrame
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        except Exception as e:
            logging.error(f"Error reading CSV for {file_name}: {e}")
            return jsonify({"error": f"Could not parse CSV file: {e}"}), 400
        
        required_columns = ['wbn', 'bag_id', 'bag_incoming_time', 'cluster_id']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": f"Missing required columns. Ensure file has: {', '.join(required_columns)}"}), 400

        df['bag_incoming_time_dt'] = pd.to_datetime(df['bag_incoming_time'], errors='coerce')
        df.dropna(subset=['bag_incoming_time_dt'], inplace=True)
        df['age_timedelta'] = datetime.now() - df['bag_incoming_time_dt']
        df['age_hours'] = df['age_timedelta'].dt.total_seconds() / 3600
        df['is_bfsi'] = df.get('is_bfsi', False).astype(str).str.lower().isin(['true', '1', 'yes'])
        df['cluster_id'] = df['cluster_id'].fillna('UNKNOWN').astype(str)

        # --- Create Bag-Level Info for Filtering ---
        bag_info = df.groupby('bag_id').agg(
            wbn_count_per_bag=('wbn', 'nunique'),
            is_bfsi_bag=('is_bfsi', 'max'),
            bag_age_hours=('age_hours', 'min'),
            bag_age_timedelta=('age_timedelta', 'min')
        ).reset_index()

        # --- Apply Filters ---
        filtered_bags = bag_info.copy()
        if min_wbn_filter is not None:
            filtered_bags = filtered_bags[filtered_bags['wbn_count_per_bag'] >= min_wbn_filter]
        if max_wbn_filter is not None:
            filtered_bags = filtered_bags[filtered_bags['wbn_count_per_bag'] <= max_wbn_filter]
        if is_bfsi_filter is not None:
            filtered_bags = filtered_bags[filtered_bags['is_bfsi_bag'] == is_bfsi_filter]
        
        if selected_age_brackets:
            conditions = []
            for bracket in selected_age_brackets:
                if bracket == "0 to 1 hr": conditions.append((filtered_bags['bag_age_hours'] >= 0) & (filtered_bags['bag_age_hours'] <= 1))
                elif bracket == "1 to 2 hr": conditions.append((filtered_bags['bag_age_hours'] > 1) & (filtered_bags['bag_age_hours'] <= 2))
                elif bracket == "2 to 2.5 hrs": conditions.append((filtered_bags['bag_age_hours'] > 2) & (filtered_bags['bag_age_hours'] <= 2.5))
                elif bracket == "More than 2.5 hrs": conditions.append((filtered_bags['bag_age_hours'] > 2.5))
            if conditions:
                filtered_bags = filtered_bags[np.logical_or.reduce(conditions)]
        
        valid_bag_ids = filtered_bags['bag_id'].unique()
        final_wbn_df = df[df['bag_id'].isin(valid_bag_ids)].copy()

        if selected_clusters:
            final_wbn_df = final_wbn_df[final_wbn_df['cluster_id'].isin(selected_clusters)]
            valid_bag_ids = final_wbn_df['bag_id'].unique()
            filtered_bags = filtered_bags[filtered_bags['bag_id'].isin(valid_bag_ids)]

        # --- Final Calculations on Filtered Data ---
        total_pending_bags = len(valid_bag_ids)
        total_pending_shipments = final_wbn_df['wbn'].nunique()

        # --- Corrected Cluster-wise Analysis ---
        cluster_details = {}
        if not final_wbn_df.empty:
            cluster_summary = final_wbn_df.groupby('cluster_id').agg(
                wbn_count=('wbn', 'nunique'),
                bag_id_count=('bag_id', 'nunique')
            ).reset_index()

            for _, row in cluster_summary.iterrows():
                cluster_id = row['cluster_id']
                bags_in_cluster_df = final_wbn_df[final_wbn_df['cluster_id'] == cluster_id]
                bag_details_for_cluster = filtered_bags[filtered_bags['bag_id'].isin(bags_in_cluster_df['bag_id'])].copy()
                wbn_counts_in_cluster_per_bag = bags_in_cluster_df.groupby('bag_id')['wbn'].nunique().reset_index(name='wbn_count_for_cluster')
                bag_details_for_cluster = pd.merge(bag_details_for_cluster, wbn_counts_in_cluster_per_bag, on='bag_id', how='left')
                bag_details_for_cluster['wbn_count_for_cluster'] = bag_details_for_cluster['wbn_count_for_cluster'].fillna(0).astype(int)

                def format_age_string(td: timedelta):
                    if pd.isna(td): return "N/A"
                    total_seconds = td.total_seconds()
                    hours, remainder = divmod(total_seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    return f"{int(hours)}h {int(minutes)}m"

                formatted_bags = [
                    {
                        "bag_id": bag['bag_id'], "wbn_count_per_bag": bag['wbn_count_per_bag'], "wbn_count_for_cluster": bag['wbn_count_for_cluster'],
                        "age_str": format_age_string(bag['bag_age_timedelta']), "age_hours": bag['bag_age_hours'], "is_bfsi": bag['is_bfsi_bag'], "cluster_id": cluster_id
                    } for bag in bag_details_for_cluster.sort_values('bag_age_hours', ascending=False).to_dict('records')
                ]

                cluster_details[cluster_id] = {
                    "bag_id_count": int(row['bag_id_count']), "wbn_count": int(row['wbn_count']), "bags": formatted_bags
                }

        # *** FIX: Restored Ageing Group Calculation Logic ***
        ageing_groups = {
            "0 to 1 hr": {"bag_id_count": 0, "wbn_count": 0, "bags": []}, "1 to 2 hr": {"bag_id_count": 0, "wbn_count": 0, "bags": []},
            "2 to 2.5 hrs": {"bag_id_count": 0, "wbn_count": 0, "bags": []}, "More than 2.5 hrs": {"bag_id_count": 0, "wbn_count": 0, "bags": []}
        }
        if not filtered_bags.empty:
            for _, row in filtered_bags.iterrows():
                age = row['bag_age_hours']
                wbn_count = row['wbn_count_per_bag']
                
                bag_data = {
                    "bag_id": row['bag_id'], "age_hours": age, "age_str": format_age_string(row['bag_age_timedelta']),
                    "wbn_count": wbn_count, "is_bfsi": row['is_bfsi_bag']
                }
                
                if 0 <= age <= 1: group = ageing_groups["0 to 1 hr"]
                elif 1 < age <= 2: group = ageing_groups["1 to 2 hr"]
                elif 2 < age <= 2.5: group = ageing_groups["2 to 2.5 hrs"]
                elif age > 2.5: group = ageing_groups["More than 2.5 hrs"]
                else: continue
                
                group["bag_id_count"] += 1
                group["wbn_count"] += wbn_count
                group["bags"].append(bag_data)

        TOTAL_ANALYSES_PERFORMED += 1
        LAST_ANALYSIS_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response_data = {
            "total_pending_bags": total_pending_bags, "total_pending_shipments": total_pending_shipments,
            "ageing_groups": ageing_groups, "cluster_details": cluster_details,
            "all_unique_clusters": sorted(df['cluster_id'].unique().tolist()),
        }

        return jsonify(response_data), 200

    except Exception as e:
        logging.error(f"An unhandled error occurred: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/status', methods=['GET'])
def get_backend_status():
    uptime_seconds = (datetime.now() - APP_START_TIME).total_seconds()
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    uptime_str = f"{int(hours)}h {int(minutes)}m"

    return jsonify({
        "status": "online", "version": BACKEND_VERSION, "uptime": uptime_str,
        "last_analysis_time": LAST_ANALYSIS_TIME, "total_analyses": TOTAL_ANALYSES_PERFORMED
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
