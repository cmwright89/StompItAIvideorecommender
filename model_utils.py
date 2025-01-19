import pandas as pd
import numpy as np
import pickle
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import product

# Define valid topics for each category and create reverse mapping
CATEGORY_TOPICS = {
    'Ski Tech': [
        'Camp Experience', 'Camp Promo', 'Carv', 'Carving', 'Gear',
        'General (ski tech)', 'Grinding', 'Jumping', 'Powdering',
        'Spinning', 'Switch', 'Touring', 'Turning', 'Trampoline',
        'Tricks', 'Fear'
    ],
    'Freestyle': [
        'Butters/Ollies', 'Camp Experience', 'Camp Promo', 'Carving',
        'Fear', 'Gear', 'General (freestyle)', 'Grabbing', 'Grinding',
        'Halfpipe', 'Inversions', 'Jibbing', 'Jumping', 'Spinning',
        'Spinning + G', 'Switch', 'Trampoline', 'Tricks', 'Turning',
        'Powdering'
    ],
    'Freeride': [
        'Butters/Ollies', 'Camp Experience', 'Camp Promo', 'Drops',
        'Gear', 'General (freeride)', 'Inversions', 'Powdering',
        'Ski Mountaineering', 'Touring', 'Trampoline', 'Turning',
        'Tricks', 'Fear', 'Grabbing', 'Jumping', 'Spinning',
        'Spinning + G', 'Switch'
    ],
    'All skiing': [
        'General (freestyle)', 'Camp Promo', 'Gear', 'General',
        'General (freeride)', 'General (ski tech)', 'Season Prep',
        'Ski Mountaineering', 'Tricks', 'Camp Experience', 'Fear',
        'Jumping', 'Powdering', 'Spinning', 'Switch', 'Trampoline',
        'Turning'
    ]
}

# Create reverse mapping of topics to categories
TOPIC_CATEGORIES = {}
for category, topics in CATEGORY_TOPICS.items():
    for topic in topics:
        if topic not in TOPIC_CATEGORIES:
            TOPIC_CATEGORIES[topic] = []
        TOPIC_CATEGORIES[topic].append(category)


def get_valid_categories(topic):
    """Get valid categories for a given topic."""
    if topic == 'All':
        return sorted(list(CATEGORY_TOPICS.keys()))
    return sorted(TOPIC_CATEGORIES.get(topic, []))


def get_valid_topics(category):
    """Get valid topics for a given category."""
    if category == 'All':
        # Return all unique topics across categories
        return sorted(list(set([
            topic for topics in CATEGORY_TOPICS.values()
            for topic in topics
        ])))
    return sorted(CATEGORY_TOPICS.get(category, []))


def load_models():
    """Load the trained models, encoders, and feature columns from pickle files."""
    try:
        with open('youtube_models.pkl', 'rb') as f:
            models = pickle.load(f)
        with open('youtube_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('youtube_feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        return models, encoders, feature_cols
    except FileNotFoundError as e:
        raise Exception("Model files not found. Please ensure all pickle files are in the current directory.") from e
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}") from e


def parse_duration(duration_str):
    """Convert ISO 8601 duration to seconds."""
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration_str)

    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return 0


def load_and_process_data(filter_date=False):
    """Load and preprocess the video data."""
    try:
        # Read the CSV file
        df = pd.read_csv('Stompit_vids.csv')

        # Convert publish date to datetime (timestamps are already tz-aware)
        df['published_date'] = pd.to_datetime(df['published_date'])
        now = pd.Timestamp.now(tz=df['published_date'].dt.tz)  # Use same timezone as data

        # Create a filtered version for predictions
        if filter_date:
            latest_date = df['published_date'].max()
            one_year_ago = latest_date - pd.Timedelta(days=365)
            df_filtered = df[df['published_date'] >= one_year_ago].copy()
        else:
            df_filtered = df.copy()

        # Process both dataframes
        for data in [df, df_filtered]:
            # Convert numeric columns
            numeric_cols = ['view_count', 'like_count', 'comment_count']
            for col in numeric_cols:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Convert duration to seconds
            data['duration_seconds'] = data['duration'].apply(parse_duration)

            # Create duration bins
            data['duration_bin'] = pd.cut(
                data['duration_seconds'],
                bins=[0, 60, 180, 300, 600, np.inf],
                labels=['0-1 min', '1-3 min', '3-5 min', '5-10 min', '10+ min']
            )

            # Calculate engagement metrics with realistic view decay pattern
            data['days_since_publish'] = (now - data['published_date']).dt.days
            data['years_since_publish'] = data['days_since_publish'] / 365

            # Define view distribution pattern (more weighted towards early period)
            def calculate_first_year_multiplier(years):
                if years <= 1:
                    return 1  # For videos less than a year old, use actual views
                else:
                    # Assume views follow a decay pattern:
                    # Year 1: 65% of total views
                    # Year 2: 20% of total views
                    # Year 3+: 15% split across remaining years
                    first_year_proportion = 0.65
                    return first_year_proportion

            data['year_multiplier'] = data['years_since_publish'].apply(calculate_first_year_multiplier)
            data['first_year_views'] = data['view_count'] * data['year_multiplier']

            data['likes_per_view'] = np.where(data['first_year_views'] > 0,
                                              data['like_count'] / data['first_year_views'],
                                              0)
            data['comments_per_view'] = np.where(data['first_year_views'] > 0,
                                                 data['comment_count'] / data['first_year_views'],
                                                 0)

            # Handle NaN values
            data.fillna({
                'first_year_views': 0,
                'like_count': 0,
                'comment_count': 0,
                'duration_seconds': 0,
                'likes_per_view': 0,
                'comments_per_view': 0
            }, inplace=True)

        return df, df_filtered

    except FileNotFoundError:
        raise Exception("Data file not found. Please ensure 'Stompit_vids.csv' is in the current directory.")
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")


def prepare_features(df, selected_params, encoders, feature_cols):
    """Prepare features for ML prediction."""
    try:
        # Handle 'All' selections with valid combinations
        if selected_params['category'] != 'All':
            valid_topics = CATEGORY_TOPICS[selected_params['category']]
            if selected_params['main topic'] == 'All':
                # Select most common valid topic for this category from historical data
                category_data = df[df['category'] == selected_params['category']]
                valid_category_data = category_data[
                    category_data['main topic'].isin(valid_topics)
                ]
                selected_params['main topic'] = valid_category_data['main topic'].mode().iloc[0]

        # Calculate format performance metrics using recent data
        format_performance = df.groupby('format').agg({
            'first_year_views': 'mean',
            'likes_per_view': 'mean',
            'comments_per_view': 'mean'
        }).reset_index()

        # Replace remaining 'All' values with most common values
        for param in ['format', 'level']:
            if selected_params[param] == 'All':
                if selected_params['category'] != 'All':
                    # Get most common value within the selected category
                    category_data = df[df['category'] == selected_params['category']]
                    selected_params[param] = category_data[param].mode().iloc[0]
                else:
                    selected_params[param] = df[param].mode().iloc[0]

        # Create interaction features
        format_category = f"{selected_params['format']}_{selected_params['category']}"
        format_topic = f"{selected_params['format']}_{selected_params['main topic']}"

        # Verify the format_category and format_topic exist in encoders
        # If not, use the most common ones from the training data
        try:
            encoders['format_category'].transform([format_category])[0]
            encoders['format_topic'].transform([format_topic])[0]
        except ValueError:
            # Find most common valid combinations from historical data
            category_data = df[df['category'] == selected_params['category']]
            most_common_format = category_data['format'].mode().iloc[0]
            most_common_topic = category_data['main topic'].mode().iloc[0]

            format_category = f"{most_common_format}_{selected_params['category']}"
            format_topic = f"{most_common_format}_{most_common_topic}"
            selected_params['format'] = most_common_format
            selected_params['main topic'] = most_common_topic

        # Get historical metrics for selected format
        format_hist = format_performance[
            format_performance['format'] == selected_params['format']
            ].iloc[0]

        # Create feature vector
        features = pd.DataFrame({
            'category_encoded': [encoders['category'].transform([selected_params['category']])[0]],
            'format_encoded': [encoders['format'].transform([selected_params['format']])[0]],
            'main topic_encoded': [encoders['main topic'].transform([selected_params['main topic']])[0]],
            'level_encoded': [encoders['level'].transform([selected_params['level']])[0]],
            'duration_bin_encoded': [encoders['duration_bin'].transform(['3-5 min'])[0]],
            'format_category_encoded': [encoders['format_category'].transform([format_category])[0]],
            'format_topic_encoded': [encoders['format_topic'].transform([format_topic])[0]],
            'duration_seconds': [300],  # default to 5 minutes
            'view_count_format_avg': [format_hist['first_year_views']],
            'likes_per_view_format_avg': [format_hist['likes_per_view']],
            'comments_per_view_format_avg': [format_hist['comments_per_view']]
        })

        # Scale numeric features
        numeric_features = ['duration_seconds', 'view_count_format_avg',
                            'likes_per_view_format_avg', 'comments_per_view_format_avg']
        features[numeric_features] = encoders['scaler'].transform(features[numeric_features])

        return features[feature_cols]

    except Exception as e:
        raise Exception(f"Error preparing features: {str(e)}")


def get_predictions(features, models):
    """Get predictions from all models."""
    try:
        # Get predictions from each model
        total_views_pred = np.expm1(models['views'].predict(features)[0])  # Reverse log transformation

        # Convert to first year views (65% of total views)
        first_year_views_pred = total_views_pred * 0.65

        # Apply a more aggressive calibration factor
        # Start reducing predictions above 25k (historical average)
        # Apply stronger reduction factors
        calibration_factor = 1.0
        if first_year_views_pred > 25000:
            # More aggressive reduction for predictions over historical average
            reduction = (first_year_views_pred - 25000) / 50000  # Steeper reduction
            calibration_factor = max(0.2, 1.0 - reduction)  # Allow reduction to 20%

        calibrated_views = first_year_views_pred * calibration_factor

        # Cap the maximum prediction at 4x the historical average
        max_views = 25000 * 4  # Cap at 100k views
        calibrated_views = min(calibrated_views, max_views)

        likes_ratio_pred = models['likes_ratio'].predict(features)[0]
        comments_ratio_pred = models['comments_ratio'].predict(features)[0]

        # Return predictions as a dictionary
        return {
            'views': calibrated_views,
            'likes_ratio': likes_ratio_pred * 100,  # Convert to percentage
            'comments_ratio': comments_ratio_pred * 100  # Convert to percentage
        }

    except Exception as e:
        raise Exception(f"Error making predictions: {str(e)}")


def analyze_optimal_duration(df, params):
    """Analyze historical data to recommend optimal video duration."""
    filtered_df = df.copy()

    # Filter by parameters if they're not 'All'
    for param, value in params.items():
        if value != 'All':
            filtered_df = filtered_df[filtered_df[param] == value]

    if len(filtered_df) == 0:
        return {
            'optimal_duration': '3-5 min',  # default if no data
            'duration_min': 180,
            'duration_max': 300,
            'confidence': 'low'
        }

    # Group by duration bin and calculate mean views
    duration_performance = filtered_df.groupby('duration_bin')['first_year_views'].agg([
        'mean', 'count'
    ]).reset_index()

    # Only consider bins with enough data (at least 2 videos)
    duration_performance = duration_performance[duration_performance['count'] >= 2]

    if len(duration_performance) == 0:
        return {
            'optimal_duration': '3-5 min',
            'duration_min': 180,
            'duration_max': 300,
            'confidence': 'low'
        }

    # Find the best performing duration bin
    best_duration = duration_performance.loc[duration_performance['mean'].idxmax(), 'duration_bin']

    # Map duration bin to minute ranges
    duration_ranges = {
        '0-1 min': (0, 60),
        '1-3 min': (60, 180),
        '3-5 min': (180, 300),
        '5-10 min': (300, 600),
        '10+ min': (600, 900)  # Cap at 15 minutes
    }

    # Determine confidence based on sample size
    sample_size = duration_performance.loc[duration_performance['duration_bin'] == best_duration, 'count'].iloc[0]
    confidence = 'high' if sample_size >= 5 else 'medium' if sample_size >= 3 else 'low'

    duration_min, duration_max = duration_ranges[best_duration]

    return {
        'optimal_duration': best_duration,
        'duration_min': duration_min,
        'duration_max': duration_max,
        'confidence': confidence
    }


def get_top_recommendations(df_filtered, selected_params, models, encoders, feature_cols, n_recommendations=3):
    """Generate top n video parameter combinations and their predictions based on user-selected parameters."""
    recommendations = []

    # Determine which parameters are fixed vs flexible (where user selected 'All')
    flexible_params = {k: v for k, v in selected_params.items() if v == 'All'}
    fixed_params = {k: v for k, v in selected_params.items() if v != 'All'}

    # Get possible values for flexible parameters, considering topic-category relationships
    param_options = {
        'format': sorted([x for x in df_filtered['format'].unique() if str(x) != 'nan' and x != 'NA']),
        'level': sorted([x for x in df_filtered['level'].unique() if str(x) != 'nan' and x != 'NA'])
    }

    # Handle category options based on selected topic
    if 'main topic' in fixed_params and 'category' in flexible_params:
        param_options['category'] = get_valid_categories(fixed_params['main topic'])
    else:
        param_options['category'] = sorted(
            [x for x in df_filtered['category'].unique() if str(x) != 'nan' and x != 'NA'])

    # Function to get valid topics based on category
    def get_topics_for_category(cat):
        if cat == 'All':
            # Get all unique topics across categories
            return sorted(list(set([topic for topics in CATEGORY_TOPICS.values() for topic in topics])))
        return get_valid_topics(cat)

    # Generate combinations of flexible parameters
    # Build lists of values to try for each flexible parameter
    param_values = {}
    for param in flexible_params:
        if param == 'main topic':
            # If category is fixed, use its topics. If category is flexible, we'll handle topics separately
            if 'category' in fixed_params:
                param_values[param] = get_topics_for_category(fixed_params['category'])
        else:
            param_values[param] = param_options.get(param, [])

    # Generate all combinations of flexible parameters
    if param_values:
        keys, values = zip(*param_values.items())
        for combination in product(*values):
            params = fixed_params.copy()
            params.update(dict(zip(keys, combination)))

            # Handle topic selection if needed
            if 'main topic' in flexible_params and 'main topic' not in params:
                category = params.get('category', fixed_params.get('category'))
                topics = get_topics_for_category(category)
                for topic in topics:
                    final_params = params.copy()
                    final_params['main topic'] = topic

                    try:
                        # Prepare features and get predictions
                        features = prepare_features(df_filtered, final_params, encoders, feature_cols)
                        predictions = get_predictions(features, models)
                        # Get duration recommendation
                        duration_recommendation = analyze_optimal_duration(df_filtered, final_params)

                        recommendations.append({
                            'parameters': final_params,
                            'predictions': predictions,
                            'duration': duration_recommendation
                        })
                    except Exception as e:
                        continue  # Skip invalid combinations
            else:
                try:
                    # Prepare features and get predictions
                    features = prepare_features(df_filtered, params, encoders, feature_cols)
                    predictions = get_predictions(features, models)
                    # Get duration recommendation
                    duration_recommendation = analyze_optimal_duration(df_filtered, params)

                    recommendations.append({
                        'parameters': params,
                        'predictions': predictions,
                        'duration': duration_recommendation
                    })
                except Exception as e:
                    continue  # Skip invalid combinations
    else:
        # If no flexible parameters, just predict for the fixed parameters
        features = prepare_features(df_filtered, fixed_params, encoders, feature_cols)
        predictions = get_predictions(features, models)
        duration_recommendation = analyze_optimal_duration(df_filtered, fixed_params)
        recommendations.append({
            'parameters': fixed_params,
            'predictions': predictions,
            'duration': duration_recommendation
        })

    # Sort by predicted views and get top n
    recommendations.sort(key=lambda x: x['predictions']['views'], reverse=True)
    return recommendations[:n_recommendations]