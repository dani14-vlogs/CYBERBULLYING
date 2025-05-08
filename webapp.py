import streamlit as st
from PIL import Image
import os
import pandas as pd
from datetime import datetime
import pickle
from functions import preprocess, custom_input_prediction
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Constants
IMAGE_DIR = "images"
DATA_DIR = "data"
MODEL_DIR = "models"
CATEGORY_IMAGES = {
    "Age": "age_cyberbullying.png",
    "Ethnicity": "ethnicity_cyberbullying.png",
    "Gender": "gender_cyberbullying.png",
    "Not Cyberbullying": "not_cyberbullying.png",
    "Other Cyberbullying": "other_cyberbullying.png",
    "Religion": "religion_cyberbullying.png"
}

# Initialize session state
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = []
if 'dataset_stats' not in st.session_state:
    st.session_state.dataset_stats = {}

# Utility functions
def load_image(image_name):
    try:
        return Image.open(os.path.join(IMAGE_DIR, image_name))
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(os.path.join(DATA_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

def analyze_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        stats = {
            'total_tweets': len(df),
            'categories': df['cyberbullying_type'].value_counts().to_dict(),
            'sample_tweets': df.head(5).to_dict('records')
        }
        return stats
    except Exception as e:
        st.error(f"Error analyzing dataset: {str(e)}")
        return None

# Admin functions
def admin_panel():
    st.sidebar.header("Admin Panel")
    
    with st.expander("🚀 Training Module"):
        st.subheader("Dataset Management")
        uploaded_file = st.file_uploader("Upload new dataset (CSV)", type=['csv'])
        
        if uploaded_file:
            if st.button("Process Dataset"):
                if save_uploaded_file(uploaded_file):
                    st.success(f"Dataset {uploaded_file.name} saved successfully!")
                    stats = analyze_dataset(os.path.join(DATA_DIR, uploaded_file.name))
                    if stats:
                        st.session_state.dataset_stats = stats
                        st.json(stats)
        
        st.subheader("Model Training")
        if st.button("Train Classification Model"):
            with st.spinner("Training model with 6 categories..."):
                try:
                    # This would normally call your training code from classifier_model.ipynb
                    # For demo purposes, we'll simulate training
                    st.session_state.performance_data.append({
                        'timestamp': datetime.now(),
                        'action': 'model_training',
                        'accuracy': 0.92,
                        'dataset': uploaded_file.name if uploaded_file else 'default'
                    })
                    st.success("Model trained successfully!")
                    st.session_state.model_trained = True
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    with st.expander("📊 Testing Module"):
        st.subheader("Performance Analysis")
        if st.button("Generate Confusion Matrix"):
            try:
                # Simulate confusion matrix - in real app, use actual predictions
                y_true = [0, 1, 2, 3, 4, 5] * 10
                y_pred = [0, 1, 2, 3, 4, 5] * 9 + [5,4,3,2,1,0]
                cm = confusion_matrix(y_true, y_pred)
                
                fig, ax = plt.subplots(figsize=(10,8))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                            display_labels=["Age", "Ethnicity", "Gender", 
                                                          "Not Cyber", "Other", "Religion"])
                disp.plot(ax=ax)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error generating matrix: {str(e)}")
        
        st.subheader("System Metrics")
        if st.session_state.performance_data:
            metrics_df = pd.DataFrame(st.session_state.performance_data)
            st.dataframe(metrics_df)
            
            # Show basic stats
            if 'accuracy' in metrics_df.columns:
                st.metric("Average Accuracy", 
                         f"{metrics_df['accuracy'].mean()*100:.2f}%")
        else:
            st.info("No performance data available")

# User functions
def user_interface():
    logo = load_image('logo.png')
    if logo:
        st.image(logo, use_column_width=True)

    st.write('''
    # Cyberbullying Tweet Recognition App
    ***
    ''')
    
    # Create tabs for different analysis types
    tab1, tab2 = st.tabs(["Single Tweet Analysis", "Bulk Analysis"])
    
    with tab1:
        st.header("Single Tweet Analysis")
        tweet_input = st.text_area("Enter tweet text:", height=150, 
                                  placeholder="Type or paste a tweet here...")
        
        if st.button("Analyze Tweet") and tweet_input:
            with st.spinner('Analyzing...'):
                try:
                    prediction = custom_input_prediction(tweet_input)
                    
                    st.session_state.performance_data.append({
                        'timestamp': datetime.now(),
                        'action': 'tweet_prediction',
                        'prediction': prediction,
                        'text_sample': tweet_input[:50] + "..."
                    })
                    
                    if prediction in CATEGORY_IMAGES:
                        image = load_image(CATEGORY_IMAGES[prediction])
                        if image:
                            st.image(image, use_column_width=True)
                        st.success(f"Prediction: {prediction}")
                    else:
                        st.error("Unexpected prediction result")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with tab2:
        st.header("Bulk Analysis")
        uploaded_file = st.file_uploader("Upload tweets file", type=['csv', 'txt'])
        
        if uploaded_file and st.button("Analyze in Bulk"):
            with st.spinner("Processing bulk data..."):
                try:
                    # Simulate bulk processing
                    sample_results = [
                        {"tweet": "Example tweet 1...", "prediction": "Age"},
                        {"tweet": "Example tweet 2...", "prediction": "Not Cyberbullying"},
                        {"tweet": "Example tweet 3...", "prediction": "Religion"}
                    ]
                    
                    st.session_state.performance_data.append({
                        'timestamp': datetime.now(),
                        'action': 'bulk_analysis',
                        'file': uploaded_file.name,
                        'count': len(sample_results)
                    })
                    
                    st.success("Bulk analysis completed!")
                    st.json({"sample_results": sample_results[:3]})
                    
                except Exception as e:
                    st.error(f"Bulk analysis failed: {str(e)}")

    # Group analysis section
    st.header("Group Analysis")
    if st.button("Detect Group Tendencies"):
        with st.spinner("Analyzing group behavior..."):
            try:
                # Simulate group analysis
                tendencies = {
                    "most_common_category": "Religion (32%)",
                    "emerging_pattern": "Age-related bullying increasing",
                    "high_risk_users": 4,
                    "recent_trends": {
                        "last_week": {"Age": 15, "Ethnicity": 22, "Religion": 32},
                        "this_week": {"Age": 28, "Ethnicity": 18, "Religion": 29}
                    }
                }
                
                st.write("""
                ### Tendency Analysis Results:
                - **Most common category**: Religion (32%)
                - **Emerging pattern**: Age-related bullying increasing
                - **High-risk users**: 4 detected
                """)
                
                st.line_chart(pd.DataFrame(tendencies['recent_trends']))
                
            except Exception as e:
                st.error(f"Tendency analysis failed: {str(e)}")

# Main app
def main():
    # Create necessary directories
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Admin toggle
    if st.sidebar.checkbox("Admin Mode"):
        st.session_state.admin_mode = True
        admin_panel()
    else:
        st.session_state.admin_mode = False
        user_interface()

    # About section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this system:**
    - 6-category classification
    - Real-time analysis
    - Group tendency detection
    - Admin training interface
    """)

if __name__ == "__main__":
    main()