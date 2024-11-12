# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

import warnings
# Suppress warnings
warnings.filterwarnings('ignore')

#######################
# Page configuration
st.set_page_config(
    page_title="Dashboard Template", # Replace this with your Project's Title
    page_icon="assets/icon.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

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

    # Sidebar Title (Change this with your project's title)
    st.title('Dashboard Template')

    # Page Button Navigation
    st.subheader("Pages")

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
    st.markdown("1. Elon Musk\n2. Jeff Bezos\n3. Sam Altman\n4. Mark Zuckerberg")

#######################
# Data

# Load data
sleep_df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Importing models

dt_classifier = joblib.load('assets/models/decision_tree_model.joblib')
#rfr_classifier = joblib.load('assets/models/random_forest_regressor.joblib')

features = ['BMICategory_Num', 'BloodPressure_Num']
disorder_list = ['Insomnia', 'None', 'Sleep-Apnea	']

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    # Your content for the ABOUT page goes here

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.markdown("""

    ZEUS
                
    `Link:` https://www.kaggle.com/datasets/arshid/iris-flower-dataset            
                
    """)

    col_iris = st.columns((3, 3, 3), gap='medium')

# Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(sleep_df, use_container_width=True, hide_index=True)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(sleep_df.describe(), use_container_width=True)

    st.markdown("""

   ZEUS      
    """)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.dataframe(sleep_df.head(), use_container_width=True, hide_index=True)

    st.markdown("""

    Since the distribution of Iris species in our dataset is **balanced** and there are **0 null values** as well in our dataset. We will be proceeding already with creating the **Embeddings** for the *species* column and **Train-Test split** for training our machine learning model.
         
    """)

    # Replace NaN values with "None" in the 'Sleep Disorder' column
    sleep_df['Sleep Disorder'] = sleep_df['Sleep Disorder'].fillna("None")

    # Pie chart for the sleep disorder column
    plt.clf()
    def pie_chart_summary():
        disorder_counts = sleep_df['Sleep Disorder'].value_counts()
        labels = disorder_counts.index

    # Plot pie chart
        plt.pie(disorder_counts, labels=labels, autopct='%1.1f%%')
        plt.title('Pie Chart of Sleep Disorder')
        st.pyplot(plt)

    pie_chart_summary()



    #Count the instances of each value in sleep disorder (label)
    label_counts = sleep_df[['Sleep Disorder']].value_counts()
    #Find the minimum instances among the three values to use as the target count
    min_count = label_counts.min()
    #Sample each category to match the minimum count
    sleep_df3 = (
        sleep_df
        .groupby('Sleep Disorder', as_index=False)
        .apply(lambda x: x.sample(min_count))
        .reset_index(drop=True)
    )

    # Pie chart for the sleep disorder column
    plt.clf()
    def pie_chart_summary1():
        disorder_counts = sleep_df3['Sleep Disorder'].value_counts()
        labels = disorder_counts.index

    # Plot pie chart
        plt.pie(disorder_counts, labels=labels, autopct='%1.1f%%')
        plt.title('Pie Chart of Sleep Disorder')
        st.pyplot(plt)
    pie_chart_summary1()

    encoder = LabelEncoder()
    sleep_df3['Gender_Num'] = encoder.fit_transform(sleep_df3['Gender'])

    sleep_df3['Occupation_Num'] = encoder.fit_transform(sleep_df3['Occupation'])

    sleep_df3['BMICategory_Num'] = encoder.fit_transform(sleep_df3['BMI Category'])

    sleep_df3['BloodPressure_Num'] = encoder.fit_transform(sleep_df3['Blood Pressure'])

    sleep_df3['SleepDisorder_Num'] = encoder.fit_transform(sleep_df3['Sleep Disorder'])

    # Mapping of the Gender and their encoded equivalent
    categorical_col = sleep_df3['Gender'].unique()
    encoded_col = sleep_df3['Gender_Num'].unique()

    # Create a new DataFrame
    gender_mapping_df = pd.DataFrame({'Gender': categorical_col, 'Gender_Num': encoded_col})

    # Display the DataFrame
    gender_mapping_df

    # Mapping of the Occupation and their encoded equivalent
    categorical_col = sleep_df3['Occupation'].unique()
    encoded_col = sleep_df3['Occupation_Num'].unique()

    # Create a new DataFrame
    occupation_mapping_df = pd.DataFrame({'Occupation': categorical_col, 'Occupation_Num': encoded_col})

    # Display the DataFrame
    occupation_mapping_df

    # Mapping of the BMI Category and their encoded equivalent
    categorical_col = sleep_df3['BMI Category'].unique()
    encoded_col = sleep_df3['BMICategory_Num'].unique()

    # Create a new DataFrame
    bmi_mapping_df = pd.DataFrame({'BMI Category': categorical_col, 'BMICategory_Num': encoded_col})

    # Display the DataFrame
    bmi_mapping_df

    # Mapping of the BP and their encoded equivalent
    categorical_col = sleep_df3['Blood Pressure'].unique()
    encoded_col = sleep_df3['BloodPressure_Num'].unique()

    # Create a new DataFrame
    bp_mapping_df = pd.DataFrame({'Blood Pressure': categorical_col, 'BloodPressure_Num': encoded_col})

    # Display the DataFrame
    bp_mapping_df

    # Mapping of the Sleep Disorder and their encoded equivalent
    categorical_col = sleep_df3['Sleep Disorder'].unique()
    encoded_col = sleep_df3['SleepDisorder_Num'].unique()

    # Create a new DataFrame
    sleepdisorder_mapping_df = pd.DataFrame({'Sleep Disorder': categorical_col, 'SleepDisorder_Num': encoded_col})

    # Display the DataFrame
    sleepdisorder_mapping_df

    sleep_df3.drop('Person ID', axis=1, inplace=True)

    st.subheader("Train-Test Split")

    sleep_df3.head()
    # Select features and target variable
    features = ['BMICategory_Num', 'BloodPressure_Num']
    X = sleep_df3[features]
    y = sleep_df3['SleepDisorder_Num']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.code("""

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    
    """)
    
    st.subheader("X_train")
    st.dataframe(X_train, use_container_width=True, hide_index=True)

    st.subheader("X_test")
    st.dataframe(X_test, use_container_width=True, hide_index=True)

    st.subheader("y_train")
    st.dataframe(y_train, use_container_width=True, hide_index=True)

    st.subheader("y_test")
    st.dataframe(y_test, use_container_width=True, hide_index=True)