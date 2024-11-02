# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

#######################
# Page configuration
st.set_page_config(
    page_title="Project Proposal Dashboard",
    page_icon="📊", # Customize with a suitable icon or emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:
    st.title('Project Proposal Dashboard')
    st.subheader("Pages")

    # Page Button Navigation
    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. Julianna Chanel Boado\n2. Kobe Lituañas\n3. Zeus Jebril A. Kubaron\n4. John Augustine Caluag\n5. Joaquin Xavier Lajom")

#######################
# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.write("This dashboard showcases our project proposal, including data cleaning, machine learning, and prediction.")

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.write("Upload and explore your dataset here.")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state.uploaded_file
        data = pd.read_csv(uploaded_file)
        
        col = st.columns((1.5, 4.5, 2), gap='medium')
        
        with col[0]:
            st.markdown('#### Graphs Column 1')
            # Example of basic summary stats or charts
            
        with col[1]:
            st.markdown('#### Graphs Column 2')
            # Main EDA visualizations, e.g., histograms or scatter plots
            
        with col[2]:
            st.markdown('#### Graphs Column 3')
            # Additional charts as needed

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")
    
    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state.uploaded_file
        st.write("Dataset Preview:")
        data = pd.read_csv(uploaded_file)
        st.write(data)

        # Cleaning Options
        if st.button("Remove Null Values"):
            data = data.dropna()
            st.write("Data after removing null values:")
            st.write(data)

        if st.button("Remove Duplicates"):
            data = data.drop_duplicates()
            st.write("Data after removing duplicates:")
            st.write(data)

        # Summary Statistics
        if st.button("Show Summary Statistics"):
            st.write("Summary Statistics:")
            st.write(data.describe())

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    if 'uploaded_file' in st.session_state:  # Check if the uploaded file exists in session state
        uploaded_file = st.session_state.uploaded_file
        data = pd.read_csv(uploaded_file)
        target = st.selectbox("Select Target Variable", options=data.columns)
        model_type = st.selectbox("Select Model", ["Random Forest", "K-Means Clustering"])

        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 100, 50)
            max_depth = st.slider("Max Depth", 2, 20, 5)

            if st.button("Train Random Forest Model"):
                X = data.drop(columns=[target])
                y = data[target]
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(X, y)
                st.write("Model trained successfully!")
                st.write("Feature Importances:", model.feature_importances_)

        elif model_type == "K-Means Clustering":
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)

            if st.button("Train K-Means Model"):
                X = data.drop(columns=[target])
                model = KMeans(n_clusters=n_clusters)
                model.fit(X)
                st.write("Model trained successfully!")
                st.write("Cluster Centers:", model.cluster_centers_)

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    st.write("This page will handle predictions based on trained models.")

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.write("Summarize the insights and outcomes of your analysis here.")
