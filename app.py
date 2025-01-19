import streamlit as st
import pandas as pd
import numpy as np
from model_utils import (
    load_models, prepare_features, get_predictions,
    load_and_process_data, get_valid_topics, get_top_recommendations
)


def reset_filters():
    """Reset all filters to default values."""
    for key in ["category", "format", "topic", "level"]:
        st.session_state[key] = "All"


def main():
    st.title("Stomp It's Video Parameter Recommendation Tool")
    st.write("Select your preferences to get ML-based recommendations for your next channel video")

    try:
        # Load data and models
        df, df_filtered = load_and_process_data()
        models, encoders, feature_cols = load_models()

        # Initialize session state for filters
        if "category" not in st.session_state:
            st.session_state["category"] = "All"
        if "format" not in st.session_state:
            st.session_state["format"] = "All"
        if "topic" not in st.session_state:
            st.session_state["topic"] = "All"
        if "level" not in st.session_state:
            st.session_state["level"] = "All"

        # Sidebar filters
        st.sidebar.header("Video Parameters")

        # Get unique values for each field and filter out nan values
        unique_categories = sorted([x for x in df['category'].unique() if str(x) != 'nan' and x != 'NA'])
        unique_formats = sorted([x for x in df['format'].unique() if str(x) != 'nan' and x != 'NA'])
        unique_levels = sorted([x for x in df['level'].unique() if str(x) != 'nan' and x != 'NA'])

        # Add category filter first
        selected_category = st.sidebar.selectbox('Category', ['All'] + unique_categories, key="category")

        # Get valid topics based on selected category
        valid_topics = ['All'] + get_valid_topics(selected_category)

        # Add remaining filters
        selected_format = st.sidebar.selectbox('Format', ['All'] + unique_formats, key="format")
        selected_topic = st.sidebar.selectbox('Main Topic', valid_topics, key="topic")
        selected_level = st.sidebar.selectbox('Level', ['All'] + unique_levels, key="level")

        # Add buttons
        show_recommendations = st.sidebar.button("Get AI Recommendations")
        reset_filters_button = st.sidebar.button("Reset Filters", on_click=reset_filters)

        if show_recommendations:
            # Create dictionary of selected parameters
            selected_params = {
                'category': selected_category,
                'format': selected_format,
                'main topic': selected_topic,
                'level': selected_level
            }

            # Check if at least one parameter is selected
            if all(value == 'All' for value in selected_params.values()):
                st.warning("Please select at least one parameter to get recommendations.")
                return

            # Filter data based on non-'All' selections
            filtered_df = df.copy()
            for param, value in selected_params.items():
                if value != 'All':
                    filtered_df = filtered_df[filtered_df[param] == value]

            # Show predictions and statistics
            if len(filtered_df) > 0:
                st.header("Current Channel Statistics for your given Parameters")

                # Video count at the top
                st.metric("Existing Videos", f"{len(filtered_df)}")
                st.divider()

                st.write(
                    "*Note: Mean metrics include viral videos (outliers) while median values represent typical performance. The metrics below are estimates of the first-year (FY) performance of current channel videos. They are estimated using a typical YouTube view pattern where approximately 65% of a video's total views occur in the first year.*")

                # Two columns for mean vs median metrics
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Mean FY Metrics")
                    st.metric("Mean Views", f"{int(filtered_df['first_year_views'].mean()):,}")
                    st.metric("Mean Like Rate", f"{(filtered_df['likes_per_view'].mean() * 100):.2f}%")
                    st.metric("Mean Comment Rate", f"{(filtered_df['comments_per_view'].mean() * 100):.2f}%")

                with col2:
                    st.subheader("Median FY Metrics")
                    st.metric("Median Views", f"{int(filtered_df['first_year_views'].median()):,}")
                    st.metric("Median Like Rate", f"{(filtered_df['likes_per_view'].median() * 100):.2f}%")
                    st.metric("Median Comment Rate", f"{(filtered_df['comments_per_view'].median() * 100):.2f}%")

                st.divider()

            # Display ML predictions
            st.header("Top 3 AI-Recommended Video Configurations")
            st.write(
                "*These predictions are estimated using the same 65% first year view pattern as explained above.*")

            st.info(
                "📊 **How predictions work:** The AI model's predictions may differ from historical ranges because it analyzes "
                "complex patterns in your data, including how different video attributes interact with each other. "
                "The model keeps your selected parameters fixed and suggests optimal values for any parameters you left as 'All'."
            )

            # Get top 3 recommendations
            top_recommendations = get_top_recommendations(df_filtered, selected_params, models, encoders, feature_cols)

            # Show which parameters were fixed by user
            fixed_params = {k: v for k, v in selected_params.items() if v != 'All'}
            if fixed_params:
                st.write("**Your Selected Parameter(s):**")
                for param, value in fixed_params.items():
                    st.write(f"- {param}: {value}")

            # Display each recommendation in a tab
            tabs = st.tabs(["Option 1", "Option 2", "Option 3"])

            for i, (tab, rec) in enumerate(zip(tabs, top_recommendations)):
                with tab:
                    st.subheader(f"Recommended Configuration {i + 1}")

                    # Show predictions in two rows
                    # First row for engagement metrics
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        first_year_views = int(rec['predictions']['views'])  # Views are already first-year predictions
                        st.metric("Predicted FY Views", f"{first_year_views:,}")
                    with metrics_cols[1]:
                        st.metric("Predicted FY Like Rate", f"{rec['predictions']['likes_ratio']:.2f}%")
                    with metrics_cols[2]:
                        st.metric("Predicted FY Comment Rate", f"{rec['predictions']['comments_ratio']:.2f}%")

                    # Second row for duration with more space
                    duration_col1, duration_col2 = st.columns([1, 2])
                    with duration_col1:
                        duration_text = f"{int(rec['duration']['duration_min'] / 60)}-{int(rec['duration']['duration_max'] / 60)} min"
                        confidence_emoji = "🎯" if rec['duration']['confidence'] == 'high' else "📊" if rec['duration'][
                                                                                                          'confidence'] == 'medium' else "💡"
                        st.metric(
                            "Recommended Length",
                            f"{duration_text} {confidence_emoji}",
                            help="Based on historical performance. Emoji indicates confidence level: 🎯=High, 📊=Medium, 💡=Low"
                        )

                    # Show recommended parameters (only those that were flexible)
                    st.write("**Recommended Values for Flexible Parameters:**")
                    for param, value in rec['parameters'].items():
                        if param not in fixed_params:
                            st.write(f"- {param}: {value}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check that all required files are present and properly formatted.")


if __name__ == "__main__":
    main()