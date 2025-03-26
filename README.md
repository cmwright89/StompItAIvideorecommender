# Stomp It Video Recommender

A machine learning-powered recommendation tool that helps predict the best video parameters for Stomp It Tutorial's Youtube content creation.

## 📋 Overview

This Streamlit application helps Stomp It Tutorials optimize their YouTube video parameters by leveraging machine learning predictions based on historical performance data. The tool analyzes past video metrics and suggests optimal combinations of video parameters (category, format, topic, level) to maximize views, likes, and engagement.

## ✨ Features

- **Parameter Selection**: Choose at least one video parameter (category, format, topic, level) to get AI recommendations for the missing parameter(s)
- **Historical Statistics**: View mean and median metrics for videos matching your selected parameter(s)
- **AI Recommendations**: Get top 3 AI-recommended video configurations with predictions for:
  - First-year views
  - Like rate
  - Comment rate
  - Optimal video duration
- **Intelligent Filtering**: Topics are automatically filtered based on selected categories for valid combinations

## 🔧 Technical Details

The application uses three trained machine learning models to predict different aspects of video performance:

- **Data Processing**: Handles YouTube data including view counts, likes, comments, and duration
- **Feature Engineering**: Creates interaction features between parameters (format_category, format_topic)
- **ML Models**: Three separate XG Boost models predict:
  - Views model: Predicts expected view count
  - Likes ratio model: Predicts the percentage of viewers who will like the video
  - Comments ratio model: Predicts the level of comment engagement
- **Duration Analysis**: Recommends optimal video length based on historical performance
- **Calibration**: Includes view prediction calibration to account for realistic YouTube performance patterns

## 🚀 How to Use

1. Select your desired parameter(s) from the sidebar (you must select at least one and up to 3)
2. Click "Get AI Recommendations" to see the top 3 recommended configurations
3. Review predicted metrics and recommended values for flexible parameters
4. Use "Reset Filters" to clear all selections

## 📊 Understanding Predictions

- **First Year (FY) Metrics**: Predictions are based on estimated first-year performance, assuming approximately 65% of a video's total views occur in the first year
- **Diverse Recommendations**: The system provides diverse recommendations across formats and levels
- **Confidence Indicators**: Duration recommendations include confidence levels (🎯=High, 📊=Medium, 💡=Low)

## 📁 Project Structure

- `app.py`: Main Streamlit application code
- `model_utils.py`: Utility functions for data processing and ML predictions
- `Stompit_vids.csv`: Historical video data (required)
- `youtube_models.pkl`: Trained ML models
- `youtube_encoders.pkl`: Feature encoders
- `youtube_feature_cols.pkl`: Feature column definitions

## ⚙️ Installation & Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install streamlit pandas numpy scikit-learn
   ```
3. Ensure all required files are in the project directory
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## 🔒 Requirements

- Python 3.6+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle

## 📝 Notes

- The prediction models are pre-trained and loaded from pickle files
- Category and topic relationships are predefined in the code
- View predictions include calibration factors to provide realistic expectations

---

*Created for Stomp It YouTube channel content optimization*
