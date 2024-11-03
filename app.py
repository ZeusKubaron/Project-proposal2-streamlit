# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#######################
# Page configuration
st.set_page_config(
    page_title="Project Proposal Dashboard",
    page_icon="📊",
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
    st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',))
    st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',))
    st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',))
    st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',))
    st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',))
    st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',))
    st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',))

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
    
    # Only upload if it's not already in session state
    if 'data' not in st.session_state:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.write(st.session_state.data)
    else:
        st.write("Dataset Preview:")
        st.write(st.session_state.data)

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
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
    
    if 'data' in st.session_state:
        data = st.session_state.data

        # Display the original dataset for reference
        st.write("Original Dataset:")
        st.write(data)

        # Cleaning Options
        if st.button("Remove Null Values"):
            st.session_state.data = data.dropna()
            st.write("Data after removing null values:")
            st.write(st.session_state.data)

        if st.button("Remove Duplicates"):
            st.session_state.data = data.drop_duplicates()
            st.write("Data after removing duplicates:")
            st.write(st.session_state.data)

        # Dropping unused column 'Person ID'
        if st.button("Drop 'Person ID' Column"):
            if 'Person ID' in st.session_state.data.columns:
                st.session_state.data.drop('Person ID', axis=1, inplace=True)
                st.write("Dropped 'Person ID' column as it does not contribute to predicting sleep disorders.")
            else:
                st.write("The 'Person ID' column does not exist in the dataset.")

        # Summary Statistics
        if st.button("Show Summary Statistics"):
            st.write("Summary Statistics:")
            st.write(data.describe())

        # Display cleaned data
        if st.button("Show Cleaned Data"):
            st.write("Cleaned Data:")
            st.write(st.session_state.data)

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    if 'data' in st.session_state:  # Check if data exists in session state
        data = st.session_state.data
        target = st.selectbox("Select Target Variable", options=data.columns)
        
        model_type = st.selectbox("Select Model", ["Random Forest", "K-Means Clustering", "Decision Tree"])

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

        elif model_type == "Decision Tree":
            # Enhanced Supervised Model
            new_features = ['BMICategory_Num', 'BloodPressure_Num']
            new_X = data[new_features]
            new_Y = data['SleepDisorder_Num']

            # Split the dataset into training and testing sets
            new_X_train, new_X_test, new_Y_train, new_Y_test = train_test_split(new_X, new_Y, test_size=0.3, random_state=42)

            # Train the Decision Tree Classifier
            new_dt_classifier = DecisionTreeClassifier(random_state=42)
            new_dt_classifier.fit(new_X_train, new_Y_train)

            # Model Evaluation
            if st.button("Evaluate Model"):
                y_pred = new_dt_classifier.predict(new_X_test)
                accuracy = accuracy_score(new_Y_test, y_pred)
                st.write(f'Accuracy: {accuracy * 100:.2f}%')

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    st.write("This page will handle predictions based on trained models.")

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.write("Summarize the insights and outcomes of your analysis here.")
