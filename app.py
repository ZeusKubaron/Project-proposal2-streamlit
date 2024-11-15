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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.preprocessing import MinMaxScaler
import numpy as np

import joblib

import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

#######################
# Page configuration
st.set_page_config(
    page_title="Sleep, Health & Lifestyle Dashboard",  # Replace this with your Project's Title
    page_icon="assets/icon.png",  # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if "page_selection" not in st.session_state:
    st.session_state.page_selection = "about"  # Default page


# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page


# Sidebar
with st.sidebar:
    # Sidebar Title (Change this with your project's title)
    st.title("Sleep, Health & Lifestyle Dashboard")
    st.code("""
    ∩――――――――――∩     ˗ˏˋ ★ ˎˊ˗
    ||  ∧  ﾍ  ||                       
    || (* ´ ▽`)||  < ᴳᵒᵒᵈᴺⁱᵍʰᵗ   ♡
    |ﾉ^⌒⌒づ`￣  ＼          
    (　ノ　　⌒ ヽ  ＼
    ＼　　||￣￣￣￣￣||
      ＼,ﾉ||
    """)

    

    # Page Button Navigation
    st.subheader("Pages")

    if st.button(
        "About", use_container_width=True, on_click=set_page_selection, args=("about",)
    ):
        st.session_state.page_selection = "about"

    if st.button(
        "Dataset",
        use_container_width=True,
        on_click=set_page_selection,
        args=("dataset",),
    ):
        st.session_state.page_selection = "dataset"

    if st.button(
        "Data Cleaning / Pre-processing",
        use_container_width=True,
        on_click=set_page_selection,
        args=("data_cleaning",),
    ):
        st.session_state.page_selection = "data_cleaning"

    if st.button(
        "EDA", use_container_width=True, on_click=set_page_selection, args=("eda",)
    ):
        st.session_state.page_selection = "eda"

    if st.button(
        "Machine Learning",
        use_container_width=True,
        on_click=set_page_selection,
        args=("machine_learning",),
    ):
        st.session_state.page_selection = "machine_learning"

    if st.button(
        "Prediction",
        use_container_width=True,
        on_click=set_page_selection,
        args=("prediction",),
    ):
        st.session_state.page_selection = "prediction"

    if st.button(
        "Conclusion",
        use_container_width=True,
        on_click=set_page_selection,
        args=("conclusion",),
    ):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown(
        "1. Zeus Jebril Kubaron\n2. Kobe Aniban Lituañas\n3. Juliana Chanel Boado\n4. Joaquin Xavier Lajom\n5. John Augustine Caluag"
    )

#######################
# Data

# Load data
sleep_df = pd.read_csv("data/Sleep_health_and_lifestyle_dataset.csv")

# Importing models
dt_classifier = joblib.load("assets/models/decision_tree_model.joblib")
rfr_classifier = joblib.load('assets/models/random_forest_regressor.joblib')

features = ["BMICategory_Num", "BloodPressure_Num"]
disorder_list = ["Insomnia", "None", "Sleep-Apnea	"]

#######################
def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.write("""
    # About this Dashboard

    Welcome to the Sleep Health and Lifestyle Dashboard. This dashboard provides insights into how lifestyle factors affect sleep quality and sleep disorder, along with predictive modeling based on machine learning algorithms. Below is a brief overview of each component:

    ---

    ### Dataset
    The **Sleep Health and Lifestyle Dataset** serves as the foundation of this analysis. This dataset captures various health and lifestyle indicators, helping us explore their impact on **Sleep Quality** and **Sleep Disorder**.

    ---

    ### Exploratory Data Analysis (EDA)
    **EDA (Exploratory Data Analysis)** offers a comprehensive look into sleep disorder and its relationship with lifestyle factors. Key visualizations include:
    - **Pie Charts** representing distribution of sleep disorder categories
    - **Scatter Plots** depicting relationships between lifestyle variables and sleep disorder

    These visualizations help us understand patterns and trends in the data.

    ---

    ### Data Cleaning and Pre-processing
    Our data cleaning and pre-processing steps include essential tasks such as:
    - Encoding categorical variables (e.g., sleep disorder categories)
    - Splitting the dataset into **training and testing sets**

    These steps ensure that our data is ready for effective model training and evaluation.

    ---

    ### Machine Learning Models
    We employed a mix of **unsupervised and supervised models** to analyze and predict sleep quality and sleep disorder:
    - **3 Unsupervised Models** for clustering individuals based on lifestyle and health factors
    - **2 Supervised Models** to classify sleep disorder based on health and lifestyle inputs

    ---

    ### Prediction
    On the **Prediction Page**, users can input their own values for lifestyle and health indicators. Based on these inputs, our trained models will provide predictions for sleep health. This feature allows users to explore personalized insights.

    ---

    ### Conclusion
    This section summarizes key insights and observations from the **EDA** and **model training** phases. We also highlight findings and patterns discovered throughout the analysis process, offering a well-rounded perspective on how lifestyle impacts sleep quality and sleep disorder.

    ---

    Thank you for using the Sleep Health and Lifestyle Dashboard. We hope this tool provides valuable insights and encourages positive lifestyle changes for better sleep quality.
    """)


# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")

    st.subheader("Content")
    st.markdown("""
        - **Comprehensive Sleep Metrics**: Explore sleep duration, quality, and factors influencing sleep patterns.
        - **Lifestyle Factors**: Analyze physical activity levels, stress levels, and BMI categories.
        - **Cardiovascular Health**: Examine blood pressure and heart rate measurements.
        - **Sleep Disorder Analysis**: Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.
    """)
    st.markdown("`Link:` https://www.kaggle.com/datasets/arshid/iris-flower-dataset")

    col_iris = st.columns((3, 3, 3), gap="medium")

    # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(sleep_df, use_container_width=True, hide_index=True)
    st.write(
        "This dataset contains 400 rows and 13 columns related to sleep health and lifestyle factors."
    )

    with st.expander("Details about the Sleep Disorder column"):
        st.markdown("""
            - **None**: The individual does not exhibit any specific sleep disorder.
            - **Insomnia**: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
            - **Sleep Apnea**: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.
        """)

    # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(sleep_df.describe(), use_container_width=True)

    with st.expander("Observations and Insights"):
        st.markdown("""
            - **Sleep Duration**: The average sleep duration is around 7 hours, aligning with standard sleep recommendations for adults.
            - **Quality of Sleep**: The mean sleep quality score is about 6.5, suggesting moderate overall sleep quality.
            - **Stress Levels**: The dataset reveals an average stress level of around 5.8, indicating moderate stress among participants.
            - **Physical Activity**: With a mean score of approximately 6.2, the data reflects a moderately active population sample.
            - **Daily Steps**: The average daily step count is around 8000, slightly below the commonly recommended 10,000 steps per day.
            - **Blood Pressure**: The mean blood pressure is approximately 120 mmHg, typical for a healthy adult population.

            This overview provides a comprehensive summary of the dataset, detailing key aspects of sleep health
            and lifestyle factors. With this information, you can proceed to deeper analyses or modeling to explore
            relationships between these variables, such as the impact of stress and physical activity on sleep quality.
        """)

# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    st.subheader("Checking the Dataframe")

    st.write("Previously, None values in Sleep Disorder Column were named \"NaN\", so we replaced said values with \"None\"")
    st.code("""
     Replace NaN values with "None" in the 'Sleep Disorder' column
        sleep_df["Sleep Disorder"] = sleep_df["Sleep Disorder"].fillna("None")
    """)
    st.dataframe(sleep_df, use_container_width=True, hide_index=True)  #!

    # Replace NaN values with "None" in the 'Sleep Disorder' column
    sleep_df["Sleep Disorder"] = sleep_df["Sleep Disorder"].fillna("None")  #!

    st.write("In the sleep_df dataframe, there are now 3 values and the sleep disorder column will not have any null values")

    # Pie chart for the sleep disorder column

    st.title("Plots/Charts")
    st.write("Here are the plots for our Sleep Disorder Column")
    plt.clf()

    st.code("""


    """)
    def pie_chart_summary():
        disorder_counts = sleep_df["Sleep Disorder"].value_counts()
        labels = disorder_counts.index

        # Plot pie chart
        plt.pie(disorder_counts, labels=labels, autopct="%1.1f%%")
        plt.title("Pie Chart of Sleep Disorder")
        st.pyplot(plt)


    pie_chart_summary()

    st.markdown("""
    There is a problem in the dataset. The 3 values under sleep disorder are not equally distributed.
    58.6% people in the dataset has "None" or does not have sleep disorder. 20.6% has insomnia
    while 20.9% have Sleep Apnea. The three values needs to be equal to ensure 
    the machine learning models will correctly predict which sleep disorder a person has.   

    __________________________________________
    The codes below will balance out the data
    __________________________________________
    """)

    st.code("""
    # Count the instances of each value in sleep disorder (label)
    label_counts = sleep_df[["Sleep Disorder"]].value_counts()
    # Find the minimum instances among the three values to use as the target count
    min_count = label_counts.min()
    # Sample each category to match the minimum count
    sleep_df3 = (
        sleep_df.groupby("Sleep Disorder", as_index=False)
        .apply(lambda x: x.sample(min_count))
        .reset_index(drop=True)
    )
    
    """)
    # Count the instances of each value in sleep disorder (label)
    label_counts = sleep_df[["Sleep Disorder"]].value_counts()
    # Find the minimum instances among the three values to use as the target count
    min_count = label_counts.min()
    # Sample each category to match the minimum count
    sleep_df3 = (
        sleep_df.groupby("Sleep Disorder", as_index=False)
        .apply(lambda x: x.sample(min_count))
        .reset_index(drop=True)
    )

    # Pie chart for the sleep disorder column
    st.write("After Balancing We check again:")
    plt.clf()

    def pie_chart_summary1():
        disorder_counts = sleep_df3["Sleep Disorder"].value_counts()
        labels = disorder_counts.index

        # Plot pie chart
        plt.pie(disorder_counts, labels=labels, autopct="%1.1f%%")
        plt.title("Pie Chart of Sleep Disorder")
        st.pyplot(plt)

    pie_chart_summary1()

    st.write("The dataset is now balance in the 'sleep_df3' data frame. All 3 values in the sleep disorder column are equally 33.3% ")

    st.write("""
    Next we Transform the categorical values into numerical using label encoder
    'Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder'""")

    st.code("""
    encoder = LabelEncoder()
    sleep_df3["Gender_Num"] = encoder.fit_transform(sleep_df3["Gender"])
    sleep_df3["Occupation_Num"] = encoder.fit_transform(sleep_df3["Occupation"])
    sleep_df3["BMICategory_Num"] = encoder.fit_transform(sleep_df3["BMI Category"])
    sleep_df3["BloodPressure_Num"] = encoder.fit_transform(sleep_df3["Blood Pressure"])
    sleep_df3["SleepDisorder_Num"] = encoder.fit_transform(sleep_df3["Sleep Disorder"])
            
    # Mapping of the Gender and their encoded equivalent
    categorical_col = sleep_df3["Gender"].unique()
    encoded_col = sleep_df3["Gender_Num"].unique()
    """)
    encoder = LabelEncoder()
    sleep_df3["Gender_Num"] = encoder.fit_transform(sleep_df3["Gender"])
    sleep_df3["Occupation_Num"] = encoder.fit_transform(sleep_df3["Occupation"])
    sleep_df3["BMICategory_Num"] = encoder.fit_transform(sleep_df3["BMI Category"])
    sleep_df3["BloodPressure_Num"] = encoder.fit_transform(sleep_df3["Blood Pressure"])
    sleep_df3["SleepDisorder_Num"] = encoder.fit_transform(sleep_df3["Sleep Disorder"])


    # Mapping of the Gender and their encoded equivalent
    categorical_col = sleep_df3["Gender"].unique()
    encoded_col = sleep_df3["Gender_Num"].unique()

    # Create a new DataFrame
    gender_mapping_df = pd.DataFrame(
        {"Gender": categorical_col, "Gender_Num": encoded_col}
    )
    st.write("Encoded Equivalent of Gender:")
    # Display the DataFrame
    gender_mapping_df

    st.write("We do the same for every Column, here are their Encoded equivalents:")
    # Mapping of the Occupation and their encoded equivalent
    categorical_col = sleep_df3["Occupation"].unique()
    encoded_col = sleep_df3["Occupation_Num"].unique()

    # Create a new DataFrame
    occupation_mapping_df = pd.DataFrame(
        {"Occupation": categorical_col, "Occupation_Num": encoded_col}
    )

    # Display the DataFrame
    st.write("Encoded Equivalent of Occupation:")
    occupation_mapping_df

    # Mapping of the BMI Category and their encoded equivalent
    categorical_col = sleep_df3["BMI Category"].unique()
    encoded_col = sleep_df3["BMICategory_Num"].unique()

    # Create a new DataFrame
    bmi_mapping_df = pd.DataFrame(
        {"BMI Category": categorical_col, "BMICategory_Num": encoded_col}
    )

    # Display the DataFrame
    st.write("Encoded Equivalent of BMI")
    bmi_mapping_df

    # Mapping of the BP and their encoded equivalent
    categorical_col = sleep_df3["Blood Pressure"].unique()
    encoded_col = sleep_df3["BloodPressure_Num"].unique()

    # Create a new DataFrame
    bp_mapping_df = pd.DataFrame(
        {"Blood Pressure": categorical_col, "BloodPressure_Num": encoded_col}
    )

    # Display the DataFrame
    st.write("Encoded Equivalent of Blood Pressure:")
    bp_mapping_df

    # Mapping of the Sleep Disorder and their encoded equivalent
    categorical_col = sleep_df3["Sleep Disorder"].unique()
    encoded_col = sleep_df3["SleepDisorder_Num"].unique()

    # Create a new DataFrame
    sleepdisorder_mapping_df = pd.DataFrame(
        {"Sleep Disorder": categorical_col, "SleepDisorder_Num": encoded_col}
    )

    # Display the DataFrame
    st.write("Encoded Equivalent of Sleep Disorders:")
    sleepdisorder_mapping_df

    sleep_df3.drop("Person ID", axis=1, inplace=True)

    st.subheader("Train-Test Split")

    sleep_df3.head()
    # Select features and target variable
    features = ["BMICategory_Num", "BloodPressure_Num"]
    X = sleep_df3[features]
    y = sleep_df3["SleepDisorder_Num"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Store the data in session state
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

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

    st.session_state.sleep_df3 = sleep_df3

elif st.session_state.page_selection == "eda":
    if "sleep_df3" in st.session_state:
        sleep_df3 = st.session_state["sleep_df3"]

        st.header("📈 Exploratory Data Analysis (EDA)")

        # Pie chart - Distribution of Sleep Disorder
        plt.pie(
            sleep_df3["Sleep Disorder"].value_counts(),
            labels=sleep_df3["Sleep Disorder"].unique(),
            autopct="%1.1f%%",
        )
        plt.title("Distribution of Sleep Disorder")
        st.pyplot(plt)
        st.write(
            "This pie chart illustrates the distribution of different sleep disorder categories in the dataset, showcasing the percentage share of each disorder."
        )

        # Scatter plot - Sleep Disorder and Gender
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Gender",
            hue="Gender",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Gender")
        plt.title("Scatter Plot of Gender by Sleep Disorder")
        plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This plot highlights the relationship between different genders and their reported sleep disorders, allowing you to visually inspect any gender-specific patterns in sleep disorders."
        )

        # Scatter plot - Sleep Disorder and Age
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3, x="Sleep Disorder", y="Age", hue="Age", palette="Set1"
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Age")
        plt.title("Scatter Plot of Age by Sleep Disorder")
        plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This visualization explores how age groups are distributed across various sleep disorders, helping identify any age-related trends in sleep health issues."
        )

        # Scatter plot - Sleep Disorder and Occupation
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Occupation",
            hue="Occupation",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Occupation")
        plt.title("Scatter Plot of Occupation by Sleep Disorder")
        plt.legend(title="Occupation", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This scatter plot examines the link between different occupations and sleep disorders, potentially revealing how certain job types might be associated with specific sleep issues"
        )

        # Scatter plot - Sleep Disorder and Sleep Duration
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Sleep Duration",
            hue="Sleep Duration",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Sleep Duration")
        plt.title("Scatter Plot of Sleep Duration by Sleep Disorder")
        plt.legend(title="Sleep Duration", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This plot investigates the correlation between sleep duration and different sleep disorders, providing insight into whether certain disorders are associated with shorter or longer sleep times."
        )

        # Scatter plot - Sleep Disorder and Quality of Sleep
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Quality of Sleep",
            hue="Quality of Sleep",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Quality of Sleep")
        plt.title("Scatter Plot of Quality of Sleep by Sleep Disorder")
        plt.legend(title="Quality of Sleep", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This scatter plot analyzes the reported quality of sleep for individuals with different sleep disorders, allowing you to see if certain disorders are linked to poorer sleep quality."
        )

        # Scatter plot - Sleep Disorder and Stress Level
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Stress Level",
            hue="Stress Level",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Stress Level")
        plt.title("Scatter Plot of Stress Level by Sleep Disorder")
        plt.legend(title="Stress Level", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This visualization assesses the relationship between stress levels and sleep disorders, which can help identify if stress is a potential contributing factor to specific sleep issues."
        )

        # Scatter plot - Sleep Disorder and BMI Category
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="BMI Category",
            hue="BMI Category",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("BMI Category")
        plt.title("Scatter Plot of BMI Category by Sleep Disorder")
        plt.legend(title="BMI Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This plot explores the association between BMI categories (e.g., underweight, normal, overweight) and various sleep disorders, highlighting any weight-related patterns in sleep health."
        )

        # Scatter plot - Sleep Disorder and Blood Pressure
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Blood Pressure",
            hue="Blood Pressure",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Blood Pressure")
        plt.title("Scatter Plot of Blood Pressure by Sleep Disorder")
        plt.legend(title="Blood Pressure", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This scatter plot examines how blood pressure levels correlate with different sleep disorders, potentially revealing if certain conditions are linked to higher or lower blood pressure."
        )

        # Scatter plot - Sleep Disorder and Heart Rate
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Heart Rate",
            hue="Heart Rate",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Heart Rate")
        plt.title("Scatter Plot of Heart Rate by Sleep Disorder")
        plt.legend(title="Heart Rate", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This plot investigates the connection between heart rate and sleep disorders, helping to see if irregular heart rates are prevalent in certain sleep health issues."
        )

        # Scatter plot - Sleep Disorder and Daily Steps
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sleep_df3,
            x="Sleep Disorder",
            y="Daily Steps",
            hue="Daily Steps",
            palette="Set1",
        )
        plt.xlabel("Sleep Disorder")
        plt.ylabel("Daily Steps")
        plt.title("Scatter Plot of Daily Steps by Sleep Disorder")
        plt.legend(title="Daily Steps", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(plt)
        st.write(
            "This scatter plot visualizes the relationship between physical activity (measured as daily steps) and sleep disorders, which might indicate whether activity levels impact sleep health."
        )

        # Pairwise Scatter plots - Features colored by Species
        sns.pairplot(sleep_df3, hue="Sleep Disorder", markers=["o"], palette="viridis")
        plt.suptitle("Scatter Plot Matrix of Sleep Features by Sleep Disorder", y=1.02)
        st.pyplot(plt)
        st.write(
            "This comprehensive pairplot displays multiple scatter plots for various features colored by sleep disorder categories, providing a visual overview of potential relationships between features and sleep disorders."
        )

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    if "sleep_df3" in st.session_state:
        sleep_df3 = st.session_state["sleep_df3"]

        st.header("🤖 Machine Learning")

        st.title("Unsupervised")

        st.markdown("""
        ### Demographic factors:
        - Age
        - Gender
        - Occupation

        ### Lifestyle factors:
        - Sleep Duration
        - Physical Activity Level
        - Stress Level
        - Daily Steps

        ### Cardiovascular health factors:
        - Heart Rate
        - Blood Pressure
        - BMI Category
        - Sleep Disorder
        """)
        sleep_df3 = st.session_state.get("sleep_df3")
        
        st.title("Demographic Factors")

        kmeans = KMeans(n_clusters=3, random_state=0)
        sleep_df3["Cluster_demographic"] = kmeans.fit_predict(
            sleep_df3[["Quality of Sleep", "Age", "Gender_Num", "Occupation_Num"]]
        )


        centroids = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=["Quality of Sleep", "Age", "Gender_Num", "Occupation_Num"],
        )
        

        # Display centroids in the Streamlit app
        st.write("Cluster Centroids:")
        st.write(centroids)

        # Sort clusters by the Quality of Sleep centroid value
        sorted_centroids = centroids.sort_values(by="Quality of Sleep").reset_index()

        st.write(
            "Based on the calculated center value of Quality of Sleep, the highest will be labeled Good Sleep, the lowest is Bad Sleep, and the middle centriods is the Moderate Sleep"
        )

        # Create a mapping of cluster labels based on your new assignment
        Cluster_demographic_labels = { 
            sorted_centroids.loc[0, "index"]: "Bad Sleep",        # Lowest Quality of Sleep
            sorted_centroids.loc[1, "index"]: "Moderate Sleep",  # Middle Quality of Sleep
            sorted_centroids.loc[2, "index"]: "Good Sleep",      # Highest Quality of Sleep
        }

        sleep_df3["Cluster_demographic_labels"] = sleep_df3["Cluster_demographic"].map(
            Cluster_demographic_labels
        )

        sleep_df3[
            [
                "Quality of Sleep",
                "Age",
                "Gender_Num",
                "Occupation_Num",
                "Cluster_demographic",
                "Cluster_demographic_labels",
            ]
        ]

        sns.pairplot(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Age",
                    "Gender_Num",
                    "Occupation_Num",
                    "Cluster_demographic_labels",
                ]
            ],
            hue="Cluster_demographic_labels",
            palette="viridis",
        )
        st.pyplot(plt)

        st.write(
            "Showcase the Pairwise Scatter Plots of the Demographic Factors"
        )

        ############################################
        st.title("Lifestyle Factors")

        kmeans = KMeans(n_clusters=3, random_state=0)
        sleep_df3["Cluster_lifestyle"] = kmeans.fit_predict(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Sleep Duration",
                    "Physical Activity Level",
                    "Stress Level",
                    "Daily Steps"
                ]
            ]
        )

        # Display centroids for lifestyle clustering
        centroids = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=[
                "Quality of Sleep",
                "Sleep Duration",
                "Physical Activity Level",
                "Stress Level",
                "Daily Steps"
            ],
        )
        st.write("Lifestyle Cluster Centroids:")
        st.write(centroids)

        # Sort clusters by the Quality of Sleep centroid value
        sorted_centroids_lifestyle = centroids.sort_values(by="Quality of Sleep").reset_index()

        st.write(
            "Based on the calculated center value of Quality of Sleep, the highest will be labeled Good Sleep, the lowest is Bad Sleep, and the middle centriods is the Moderate Sleep"
        )

        # Create a mapping of cluster labels based on your new assignment
        Cluster_lifestyle_labels = { 
            sorted_centroids.loc[0, "index"]: "Bad Sleep",        # Lowest Quality of Sleep
            sorted_centroids.loc[1, "index"]: "Moderate Sleep",  # Middle Quality of Sleep
            sorted_centroids.loc[2, "index"]: "Good Sleep",      # Highest Quality of Sleep
        }

        sleep_df3["Cluster_lifestyle_labels"] = sleep_df3["Cluster_lifestyle"].map(
            Cluster_lifestyle_labels
        )

        # Displaying the cluster with labels
        st.write(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Sleep Duration",
                    "Physical Activity Level",
                    "Stress Level",
                    "Daily Steps", 
                    "Cluster_lifestyle",
                    "Cluster_lifestyle_labels",
                ]
            ]
        )

        # Pairplot for lifestyle factors
        sns.pairplot(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Sleep Duration",
                    "Physical Activity Level",
                    "Stress Level",
                    "Daily Steps",
                    "Cluster_lifestyle_labels",
                ]
            ],
            hue="Cluster_lifestyle_labels",
            palette="viridis",
        )
        st.pyplot(plt)

        st.write(
            "Showcase the Pairwise Scatter Plots of the Lifestyle Factors"
        )

        ######################
        st.title("Cardiovascular Health Factors")

        kmeans = KMeans(n_clusters=3, random_state=0)
        sleep_df3["Cluster_cardiovascularH"] = kmeans.fit_predict(
            sleep_df3[
                [
                "Quality of Sleep",
                "Heart Rate",
                "BloodPressure_Num",
                "BMICategory_Num",
                "SleepDisorder_Num"
                ]
            ]
        )

        # Display centroids for cardiovascular health clustering
        centroids = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=[
                "Quality of Sleep",
                "Heart Rate",
                "BloodPressure_Num",
                "BMICategory_Num",
                "SleepDisorder_Num"
            ],
        )

        st.write("Cardiovascular Health Cluster Centroids:")
        st.write(centroids)

        # Sort clusters by the Quality of Sleep centroid value
        sorted_centroids_cardiovascularH = centroids.sort_values(by="Quality of Sleep").reset_index()

        st.write(
            "Based on the calculated center value of Quality of Sleep, the highest will be labeled Good Sleep, the lowest is Bad Sleep, and the middle centriods is the Moderate Sleep"
        )

        # Create a mapping of cluster labels based on your new assignment
        Cluster_cardiovascularH_labels = { 
            sorted_centroids.loc[0, "index"]: "Bad Sleep",        # Lowest Quality of Sleep
            sorted_centroids.loc[1, "index"]: "Moderate Sleep",  # Middle Quality of Sleep
            sorted_centroids.loc[2, "index"]: "Good Sleep",      # Highest Quality of Sleep
        }

        sleep_df3["Cluster_cardiovascularH_labels"] = sleep_df3["Cluster_cardiovascularH"
        ].map(Cluster_cardiovascularH_labels)

        sleep_df3[
            [
                "Quality of Sleep",
                "Heart Rate",
                "BloodPressure_Num",
                "BMICategory_Num",
                "SleepDisorder_Num",
                "Cluster_cardiovascularH",
                "Cluster_cardiovascularH_labels",
            ]
        ]

        sns.pairplot(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Heart Rate",
                    "BloodPressure_Num",
                    "BMICategory_Num",
                    "SleepDisorder_Num",
                    "Cluster_cardiovascularH_labels",
                ]
            ],
            hue="Cluster_cardiovascularH_labels",
            palette="viridis",
        )
        st.pyplot(plt)

        st.write(
            "Showcase the Pairwise Scatter Plots of the Cardiovascular Health Factors"
        )

        #End of unsupervised
        #################################################


        st.title("Supervised")

        # Extract unique values for the Sleep Disorder columns
        unique_sleepD = sleep_df3["Sleep Disorder"].unique()
        unique_sleepD_num = sleep_df3["SleepDisorder_Num"].unique()

        # Create a new DataFrame with the unique values side-by-side
        sleepDisorder_unique_values_df = pd.DataFrame(
            {
                "Unique Column1": unique_sleepD,
                "Unique Column2": pd.Series(unique_sleepD_num),
            }
        )

        # Display the result
        st.write("Sleep Disorder Values")
        sleepDisorder_unique_values_df

        # Define the features and target based on your selection
        features = [
            "Gender_Num",
            "Age",
            "Occupation_Num",
            "Sleep Duration",
            "Quality of Sleep",
            "Physical Activity Level",
            "Stress Level",
            "BMICategory_Num",
            "BloodPressure_Num",
            "Heart Rate",
            "Daily Steps",
        ]
        X = sleep_df3[features]
        y = sleep_df3["SleepDisorder_Num"]

        st.code("""
        # Define the features and target based on your selection
        features = ['Gender_Num', 'Age', 'Occupation_Num', 'Sleep Duration', 'Quality of Sleep',
                    'Physical Activity Level', 'Stress Level', 'BMICategory_Num', 'BloodPressure_Num',
                    'Heart Rate', 'Daily Steps']
        X = sleep_df3[features]
        y = sleep_df3['SleepDisorder_Num']
        """)

        st.write("X values")
        X
        st.write("y values")
        y
        # Store in session state if needed
        st.session_state["X_train"] = X
        st.session_state["y_train"] = y

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        st.code("""
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        """)

        # Display the shapes and head of each set in Streamlit
        st.write("**X_train Shape:**", X_train.shape)
        st.write("**X_train Head:**")
        st.write(X_train.head())

        st.write("**X_test Shape:**", X_test.shape)
        st.write("**X_test Head:**")
        st.write(X_test.head())

        st.write("**y_train Shape:**", y_train.shape)
        st.write("**y_train Head:**")
        st.write(y_train.head())

        st.write("**y_test Shape:**", y_test.shape)
        st.write("**y_test Head:**")
        st.write(y_test.head())

        st.subheader("Train the Decision Tree Classifier")
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X, y)

        st.code("""
        dt_classifier = DecisionTreeClassifier(random_state=42)
        dt_classifier.fit(X, y)
        """)

        st.subheader("Model Evaluation")
        # Evaluate the model
        y_pred = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

        st.code("""
        y_pred = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        """)

        st.subheader("Feature Importance")
        feature_importance = dt_classifier.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importance}
        )

        # Calculate the importance as a percentage
        importance_df["Importance (%)"] = (
            importance_df["Importance"] / importance_df["Importance"].sum()
        ) * 100

        # Sort the DataFrame by importance for better readability
        importance_df = importance_df.sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        # Display the resulting DataFrame
        st.write("**Feature Importance**")
        st.dataframe(importance_df)

        st.title("Enchanced Supervised Model")

        # Select new features and new target variable
        Newfeatures = ["BMICategory_Num", "BloodPressure_Num"]
        NewX = sleep_df3[Newfeatures]
        NewY = sleep_df3["SleepDisorder_Num"]

        st.code("""
        # Select new features and new target variable
        Newfeatures = ['BMICategory_Num', 'BloodPressure_Num']
        NewX = sleep_df3[Newfeatures]
        NewY = sleep_df3['SleepDisorder_Num']
        """)

        # Display the selected features and target variable
        st.subheader("Selected Features and Target Variable")
        st.write("**Features:**")
        st.dataframe(NewX)
        st.write("**Target Variable:**")
        st.dataframe(NewY)

        # Split the dataset into training and testing sets
        NewX_train, NewX_test, NewY_train, NewY_test = train_test_split(
            NewX, NewY, test_size=0.3, random_state=42
        )

        # Display shapes and sample data of the training/testing sets
        st.subheader("Training and Testing Data Overview")
        st.write("**Training Features Shape:**", NewX_train.shape)
        st.write("**Training Features Sample:**")
        st.dataframe(NewX_train.head())

        st.write("**Testing Features Shape:**", NewX_test.shape)
        st.write("**Testing Features Sample:**")
        st.dataframe(NewX_test.head())

        st.write("**Training Target Shape:**", NewY_train.shape)
        st.write("**Training Target Sample:**")
        st.dataframe(NewY_train.head())

        st.write("**Testing Target Shape:**", NewY_test.shape)
        st.write("**Testing Target Sample:**")
        st.dataframe(NewY_test.head())

        st.subheader("Train the Decision Tree Classifier")

        st.code("""
        new_dt_classifier = DecisionTreeClassifier(random_state=42)
        new_dt_classifier.fit(NewX_train, NewY_train)
        """)

        # Train the Decision Tree Classifier
        new_dt_classifier = DecisionTreeClassifier(random_state=42)
        new_dt_classifier.fit(NewX_train, NewY_train)

        st.subheader("Model Evaluation")

        # Evaluate the model
        y_pred = new_dt_classifier.predict(NewX_test)
        accuracy = accuracy_score(NewY_test, y_pred)
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

        st.code("""
        y_pred = new_dt_classifier.predict(NewX_test)
        accuracy = accuracy_score(NewY_test, y_pred)
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
        """)

        #Train the Random Forest Regressor
        st.subheader("Train the Random Forest Regressor")
        st.code("""
                    rfr_classifier = RandomForestRegressor()
                    rfr_classifier.fit(X_train, y_train)     
                """)

        st.subheader("Model Evaluation")

        st.code("""
                    train_accuracy = rfr_classifier.score(X_train, y_train) #train daTa
                    test_accuracy = rfr_classifier.score(X_test, y_test) #test daTa

                    print(f'Train Accuracy: {train_accuracy * 100:.2f}%')
                    print(f'Test Accuracy: {test_accuracy * 100:.2f}%') 
               """)

        st.write("""
                    **Train Accuracy:** 98.58%\n
                     **Test Accuracy:** 99.82%                
                """)
        st.subheader("Feature Importance")

        st.code("""
                    random_forest_feature_importance = pd.Series(rfr_classifier.feature_importances_, index=X_train.columns)
                    random_forest_feature_importance
                """)

        rfr_feature_importance = {
        'Feature': ['BMICategory_Num', 'BloodPressure_Num'],
        'Importance': [0.21843, 0.78157]
        }

        rfr_feature_importance_df = pd.DataFrame(rfr_feature_importance)

        st.dataframe(rfr_feature_importance_df, use_container_width=True, hide_index=True)

        feature_importance_plot(rfr_feature_importance_df, 500, 500, 2)

        st.markdown("""
                    Upon running `.feature_importances` in the `Random Forest Regressor Model` to check how each sleep disorder's features influence the training of our model, it is clear that **BloodPressure_Num** holds the most influence in our model's decisions having **0.78** or **78%** importance. Then **BMICategory_Num** which is far behind with **0.22** or **22%** importance.
                    """)
        
        st.subheader("Number of Trees")
        st.code("""
                    print(f"Number of trees made: {len(rfr_classifier.estimators_)}")    
                """)

        st.markdown("**Number of trees made:** 100")

        st.subheader("Plotting the Forest")

        forest_image = Image.open('assets/models/RFRForest.png')
        st.image(forest_image, caption='Random Forest Regressor - Forest Plot')

        st.markdown("This graph shows **all of the decision trees** made by our **Random Forest Regressor** model which then forms a **Forest**.")

        st.subheader("Forest - Single Tree")

        forest_single_tree_image = Image.open('assets/models/RFRSingle.png')
        st.image(forest_single_tree_image, caption='Random Forest Regressor - Single Tree')

        st.markdown("This **Tree Plot** shows a single tree from our Random Forest Regressor model.")

# Prediction Page
elif st.session_state.page_selection == "prediction":
    if "sleep_df3" in st.session_state:
        sleep_df3 = st.session_state["sleep_df3"]



        st.header("🔮 Prediction Page")

        
        # Unsupervised Part
        st.title("Unsupervised Models Prediction")

        st.markdown("Analyze and label if a person gets 'Good', 'Middle', or 'Bad' quality of sleep based on certain factors")

        col_pred0 = st.columns((1, 4), gap="medium")

        with col_pred0[0]:

            if 'clear' not in st.session_state:
                st.session_state.clear = False

            with st.expander('Options', expanded=True):
                show_dataset_unsupervised = st.checkbox('Show Dataset')
                show_classes_unsupervised = st.checkbox('Show All Classes')
                show_demoFac = st.checkbox('Show Demographic Factors')
                show_lifeFac = st.checkbox('Show Lifestyle Factors')
                show_cardioFac = st.checkbox('Show Cardiovascular Health Factors')

                clear_results_unsupervised = st.button('Clear Results', key='clear_results_unsupervised')

                if clear_results_unsupervised:
                    st.session_state.clear = True
        
        with col_pred0[1]:
            # Define the columns to display
            columnsdisplay_all = ["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", 
                                "Physical Activity Level", "Stress Level", "BMI Category", "Blood Pressure", 
                                "Heart Rate", "Daily Steps", "Sleep Disorder"]
            columnsdisplay_demoFac = ["Gender", "Age", "Occupation"]
            columnsdisplay_lifeFac = ["Sleep Duration", "Physical Activity Level", "Stress Level", "Daily Steps"]
            columnsdisplay_cardioFac = ["Heart Rate", "Blood Pressure", "BMI Category", "Sleep Disorder"]

            # Display 3 samples for each factors in unsupervised model
            
            if show_classes_unsupervised or show_demoFac:
                st.subheader("Demographic Samples")
                st.dataframe(sleep_df3[columnsdisplay_demoFac].head(10), use_container_width=True, hide_index=True)
            
            if show_classes_unsupervised or show_lifeFac:
                st.subheader("Lifestyle Samples")
                st.dataframe(sleep_df3[columnsdisplay_lifeFac].head(10), use_container_width=True, hide_index=True)

            if show_classes_unsupervised or show_cardioFac:
                st.subheader("Cardiovascular Health Samples Samples")
                st.dataframe(sleep_df3[columnsdisplay_cardioFac].head(10), use_container_width=True, hide_index=True)

            if show_dataset_unsupervised:
                st.subheader("Dataset")
                st.dataframe(sleep_df3[columnsdisplay_all], use_container_width=True, hide_index=True)

            if st.session_state.clear:
                st.write("Clearing results...")
                st.session_state.clear = False

       
##############################


    
        col_pred1 = st.columns((3, 3, 3), gap="medium")

        with col_pred1[0]:
            st.markdown("#### 👥 Demographic Factors")
    
            # Input value for demographic factors
            dt_age_demoFactor = st.number_input('Age (27 - 60)', min_value=27, max_value=60, step=1, key='dt_age_demoFactor')

            # Dropdown menus for demographic factors
            gender_values = ["Male", "Female"]
            occupation_values = ['Teacher', 
                                'Accountant', 
                                'Salesperson',
                                'Nurse', 
                                'Lawyer',
                                'Doctor',
                                'Engineer',
                                'Software Engineer',
                                'Scientist',
                                'Sales Representative']

            # Mapping Gender and Occupation to Numerical Values
            gender_mapping = {"Male": 1, "Female": 0}
            occupation_mapping = {'Teacher': 9,
                                'Accountant': 0,
                                'Salesperson': 6,
                                'Nurse': 4, 
                                'Lawyer': 3,
                                'Doctor': 1,
                                'Engineer': 2,
                                'Software Engineer': 8,
                                'Scientist': 7,
                                'Sales Representative': 5}

            # Dropdown Menus
            dt_gender_demoFactor = st.selectbox("Gender", options=gender_values, index=0, key="dt_genderage_demoFactor")
            dt_occupation_demoFactor = st.selectbox("Occupation", options=occupation_values, index=0, key="dt_occupationage_demoFactor")

            # Class labels for prediction
            classes_list_demoFactor = sleep_df3['Cluster_demographic_labels'].unique()

            # Button to Detect Sleep Quality
            if st.button("Label", key="dt_detect_demoFactor"):
                # Convert categorical inputs to numerical values
                dt_gender_num = gender_mapping[dt_gender_demoFactor]
                dt_occupation_num = occupation_mapping[dt_occupation_demoFactor]

                # Prepare the input data for prediction using encoded values
                input_data_demoFactor = [[dt_age_demoFactor, dt_gender_num, dt_occupation_num]]

                # Assuming sleep_df3 is available and preprocessed
                kmeans_demoFactor = KMeans(n_clusters=3, random_state=42)

                # Train the model (this happens every time the app runs)
                kmeans_demoFactor.fit(sleep_df3[['Age', 'Gender_Num', 'Occupation_Num']])

                # Prepare the input data for prediction (same as before)
                input_data_demoFactor = [[dt_age_demoFactor, dt_gender_num, dt_occupation_num]]

                # Make the prediction
                prediction_demoFactor = kmeans_demoFactor.predict(input_data_demoFactor)

                # Map the predicted cluster to sleep quality
                predicted_label_demoFactor = classes_list_demoFactor[prediction_demoFactor[0]]

                # Display the result
                st.markdown(f"### The label sleep quality is: `{predicted_label_demoFactor}`")
           

       
            with col_pred1[1]:
                st.markdown("#### 👥 Lifestyle Factors")

                # Input values for lifestyle factors
                dt_sleepduration_lifeFactor = st.number_input(
                    'Sleep Duration (5 - 9)', min_value=5.0, max_value=9.0, step=0.1, key='dt_sleepduration_lifeFactor')
                dt_physicalactivitylevel_lifeFactor = st.number_input(
                    'Physical Activity Level (30 - 90)', min_value=30, max_value=90, step=1, key='dt_physicalactivitylevel_lifeFactor')
                dt_stresslevel_lifeFactor = st.number_input(
                    'Stress Level (3 - 8)', min_value=3, max_value=8, step=1, key='dt_stresslevel_lifeFactor')
                dt_dailysteps_lifeFactor = st.number_input(
                    'Daily Steps (3000 - 10000)', min_value=3000, max_value=10000, step=100, key='dt_dailysteps_lifeFactor')

                # Class labels for prediction
                classes_list_lifeFactor = sleep_df3['Cluster_cardiovascularH_labels'].unique()

                    # Button to Detect Sleep Quality
                if st.button("Label", key="dt_detect_lifeFactor"):
                        # Prepare the input data for prediction
                    # Prepare the input data for prediction using encoded values
                    input_data_lifeFactor = [[dt_sleepduration_lifeFactor, dt_physicalactivitylevel_lifeFactor, dt_stresslevel_lifeFactor, dt_dailysteps_lifeFactor]]

                    # Assuming sleep_df3 is available and preprocessed
                    kmeans_lifeFactor = KMeans(n_clusters=3, random_state=42)

                    # Train the model (this happens every time the app runs)
                    kmeans_lifeFactor.fit(sleep_df3[['Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num']])

                    # Prepare the input data for prediction (same as before)
                    input_data_lifeFactor = [[dt_sleepduration_lifeFactor, dt_physicalactivitylevel_lifeFactor, dt_stresslevel_lifeFactor, dt_dailysteps_lifeFactor]]

                    # Make the prediction
                    prediction_lifeFactor = kmeans_lifeFactor.predict(input_data_lifeFactor)

                    # Map the predicted cluster to sleep quality
                    predicted_label_lifeFactor = classes_list_lifeFactor[prediction_lifeFactor[0]]

                        # Display the result
                    st.markdown(f"### The label sleep quality label is: `{predicted_label_lifeFactor}`")


        with col_pred1[2]:
            st.markdown("#### 👥 Cardiovascular health Factors")

            # Input value for demographic factors
            dt_heartrate_cardioFactor = st.number_input('Heart Rate [bpm] (65 - 86)', min_value=65, max_value=86, step=1, key='dt_heartrate_cardioFactor')


            # Dropdown menus for demographic factors
            bloodpressure_values = ['135/90',
                             '130/85',
                             '130/86',
                             '115/75',
                             '140/95',
                             '125/80',
                             '142/92',
                             '132/87',
                             '140/90',
                             '129/84',
                             '120/80',
                             '119/77',
                             '125/82',
                             '115/78',
                             '117/76',
                             '128/84',
                             '139/91',
                             '135/88',
                             '131/86']
            bmicategory_values = ['Overweight', 
                                  'Obese', 
                                  'Normal',
                                  'Normal Weight']
            sleepdisorder_values = ['None', 
                                  'Insomnia', 
                                  'Sleep Apnea']

            # Mapping Gender and Occupation to Numerical Values
            bloodpressure_mapping = {'135/90': 14,
                              '130/85': 9,
                              '130/86': 10,
                              '115/75': 0,
                              '140/95': 17,
                              '125/80': 5,
                              '142/92': 18,
                              '132/87': 12,
                              '140/90': 16,
                              '129/84': 8,
                              '120/80': 4,
                              '119/77': 3,
                              '125/82': 6,
                              '115/78': 1,
                              '117/76': 2,
                              '128/84': 7,
                              '139/91':	15,
                              '135/88':	13,
                              '131/86':	11}
            bmicategory_mapping = {'Overweight': 3, 
                                  'Obese': 2, 
                                  'Normal': 0,
                                  'Normal Weight': 1}
            sleepdisorder_mapping = {'None': 1, 
                                     'Insomnia': 0,
                                     'Sleep Apnea': 2}

            # Dropdown Menus
            dt_bloodpressure_cardioFactor = st.selectbox("Blood Pressure", options=bloodpressure_values, index=0, key="dt_bloodpressure_values")
            dt_bmicategory_cardioFactor = st.selectbox("BMI Categort", options=bmicategory_values, index=0, key="dt_bmicategory_cardioFactor")
            dt_sleepdisorder_cardioFactor = st.selectbox("Sleep Disorder", options=sleepdisorder_values, index=0, key="dt_sleepdisorder_cardioFactor")


            # Class labels for prediction
            classes_list_cardioFactor = sleep_df3['Cluster_cardiovascularH_labels'].unique()


            # Button to Detect Sleep Quality
            if st.button("Label", key="dt_detect_cardioFactor"):
                # Convert categorical inputs to numerical values
                dt_bloodpressure_num = bloodpressure_mapping[dt_bloodpressure_cardioFactor]
                dt_bmicategory_num = bmicategory_mapping[dt_bmicategory_cardioFactor]
                dt_sleepdisorder_num = sleepdisorder_mapping[dt_sleepdisorder_cardioFactor]

                # Prepare the input data for prediction using encoded values
                input_data_cardioFactor = [[dt_heartrate_cardioFactor, dt_bloodpressure_num, dt_bmicategory_num, dt_sleepdisorder_num]]

                # Assuming sleep_df3 is available and preprocessed
                kmeans_cardioFactor = KMeans(n_clusters=3, random_state=42)

                # Train the model (this happens every time the app runs)
                kmeans_cardioFactor.fit(sleep_df3[['Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num']])

                # Prepare the input data for prediction (same as before)
                input_data_cardioFactor = [[dt_heartrate_cardioFactor, dt_bloodpressure_num, dt_bmicategory_num, dt_sleepdisorder_num]]

                # Make the prediction
                prediction_cardioFactor = kmeans_cardioFactor.predict(input_data_cardioFactor)

                # Map the predicted cluster to sleep quality
                predicted_label_cardioFactor = classes_list_cardioFactor[prediction_cardioFactor[0]]

                # Display the result
                st.markdown(f"### The label sleep quality is: `{predicted_label_cardioFactor}`")
           







    
    #End of Unsupervised parts
    ##################################################################################

    #Supervised part
    st.title("Supervised Models Prediction")
    st.markdown("Detect if a person has a sleep disorder and which kind based on certain factors")


    col_pred2 = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred2[0]:
        with st.expander('Options', expanded=True):
            # Assign unique keys to each checkbox
            show_dataset_supervised = st.checkbox('Show Dataset', key='show_dataset_supervised')
            show_classes_supervised = st.checkbox('Show All Classes', key='show_classes_supervised')
            show_none = st.checkbox('Show None Samples', key='show_none')
            show_insomnia = st.checkbox('Show Insomnia Samples', key='show_insomnia')
            show_sleepApnea = st.checkbox('Show Sleep Apnea Samples', key='show_sleep_apnea')


            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:

                st.session_state.clear = True
    


    with col_pred2[1]:
        st.markdown("#### 🌲 Decision Tree Classifier")
        
        # Dropdown inputs for categorical features
        gender = st.selectbox('Gender', options=['Male', 'Female'], key='gender', index=0)
        occupation = st.selectbox('Occupation', 
                options=['Teacher', 
                          'Accountant', 
                          'Salesperson',
                          'Nurse', 
                          'Lawyer',
                          'Doctor',
                          'Engineer',
                          'Software Engineer',
                          'Scientist',
                          'Sales Representative'], 
                key='occupation', index=0)
        
        bmi_category = st.selectbox('BMI Category', 
                                    options=['Overweight', 
                                             'Obese', 
                                             'Normal'], 
                                             key='bmi_category', index=0)
        
        blood_pressure = st.selectbox('Blood Pressure', 
                                      options=['135/90',
                                               '130/85',
                                               '130/86',
                                               '115/75',
                                               '140/95',
                                               '125/80',
                                               '142/92',
                                               '132/87',
                                               '140/90',
                                               '129/84',
                                               '120/80',
                                               '119/77',
                                               '125/82',
                                               '115/78',
                                               '117/76',
                                               '128/84',
                                               '139/91',
                                               '135/88',
                                               '131/86'], 
                                               key='blood_pressure', index=0
                                    )



        # Input boxes for numerical features
        age = st.number_input('Age (27 - 60)', min_value=27, max_value=60, step=1, key='age')
        sleep_duration = st.number_input('Sleep Duration (5 - 9)', min_value=5.0, max_value=9.0, step=0.1, key='sleep_duration')
        quality_of_sleep = st.number_input('Quality of Sleep (4 - 9)', min_value=4, max_value=9, step=1, key='quality_of_sleep')
        physical_activity_level = st.number_input('Physical Activity Level (30 - 90)', min_value=30, max_value=90, step=1, key='physical_activity_level')
        stress_level = st.number_input('Stress Level (3 - 8)', min_value=3, max_value=8, step=1, key='stress_level')
        heart_rate = st.number_input('Heart Rate [bpm] (65 - 86)', min_value=65, max_value=86, step=1, key='heart_rate')
        daily_steps = st.number_input('Daily Steps (3000 - 10000)', min_value=3000, max_value=10000, step=100, key='daily_steps')


        # Class labels for prediction
        classes_list = ["None", "Insomnia", "Sleep Apnea"]

        # Button to process or predict
        if st.button('Detect', key='dt_detect_DecisionTreeClassifier'):
            # Map categorical inputs to numeric codes 
            gender_map = {'Male': 0, 'Female': 1}
            occupation_map = {
                'Teacher': 9,
                'Accountant': 0,
                'Salesperson': 6,
                'Nurse': 4, 
                'Lawyer': 3,
                'Doctor': 1,
                'Engineer': 2,
                'Software Engineer': 8,
                'Scientist': 7,
                'Sales Representative': 5
            }
            bmi_map = {  # Fixed to a dictionary
                'Overweight': 1,
                'Obese': 2,
                'Normal': 0
            }
            bp_map = {
                '135/90': 14,
                '130/85': 9,
                '130/86': 10,
                '115/75': 0,
                '140/95': 17,
                '125/80': 5,
                '142/92': 18,
                '132/87': 12,
                '140/90': 16,
                '129/84': 8,
                '120/80': 4,
                '119/77': 3,
                '125/82': 6,
                '115/78': 1,
                '117/76': 2,
                '128/84': 7,
                '139/91': 15,
                '135/88': 13,
                '131/86': 11
            }

            # Encode categorical inputs
            Gender_Num = gender_map[gender]
            Occupation_Num = occupation_map[occupation]
            BMICategory_Num = bmi_map[bmi_category]  # Fixed
            BloodPressure_Num = bp_map[blood_pressure]

            # Prepare the input data for the model
            dt_input_data = [[
                Gender_Num, 
                age, 
                Occupation_Num, 
                sleep_duration, 
                quality_of_sleep, 
                physical_activity_level, 
                stress_level, 
                BMICategory_Num, 
                BloodPressure_Num, 
                heart_rate, 
                daily_steps
            ]]

            # Predict using the Decision Tree Classifier
            dt_prediction = dt_classifier.predict(dt_input_data)

            # Display the prediction result (adjust classes_list for your specific output classes)
            st.markdown(f'The predicted outcome is: `{classes_list[dt_prediction[0]]}`')


 
    
    with col_pred2[2]:
        st.markdown("#### 🌲🌲🌲 Random Forest Regressor")
        # Dropdown inputs for categorical features
        
        bmi_category = st.selectbox('BMI Category', 
                                    options=['Overweight', 
                                             'Obese', 
                                             'Normal'], 
                                             key='bmi_category1', index=0)
        
        blood_pressure = st.selectbox('Blood Pressure', 
                                      options=['135/90',
                                               '130/85',
                                               '130/86',
                                               '115/75',
                                               '140/95',
                                               '125/80',
                                               '142/92',
                                               '132/87',
                                               '140/90',
                                               '129/84',
                                               '120/80',
                                               '119/77',
                                               '125/82',
                                               '115/78',
                                               '117/76',
                                               '128/84',
                                               '139/91',
                                               '135/88',
                                               '131/86'], 
                                               key='blood_pressure1', index=0
                                    )

        # Class labels for prediction
        classes_list = ["None", "Insomnia", "Sleep Apnea"]

        # Button to process or predict
        if st.button('Detect', key='rfr_detect_RandomForestClassifier'):
            # Map categorical inputs to numeric codes 
            bmi_map = {'Overweight': 1, 
                       'Obese': 2, 
                       'Normal': 0}
            bp_map = {'135/90': 14,
                      '130/85': 9,
                      '130/86': 10,
                      '115/75': 0,
                      '140/95': 17,
                      '125/80': 5,
                      '142/92': 18,
                      '132/87': 12,
                      '140/90': 16,
                      '129/84': 8,
                      '120/80': 4,
                      '119/77': 3,
                      '125/82': 6,
                      '115/78': 1,
                      '117/76': 2,
                      '128/84': 7,
                      '139/91':	15,
                      '135/88':	13,
                      '131/86':	11}

            # Encode categorical inputs
            BMICategory_Num = bmi_map[bmi_category]
            BloodPressure_Num = bp_map[blood_pressure]

            # Prepare the input data for the model
            dt_input_data = [[ 
                BMICategory_Num, 
                BloodPressure_Num, 
            ]]

            # Predict using the Random Forest Regressor
            rfr_prediction = rfr_classifier.predict(dt_input_data)

            # Convert the prediction to an integer index
            predicted_index = int(round(rfr_prediction[0]))  # Round and cast to integer

            # Ensure the predicted index is within bounds of the classes_list
            if 0 <= predicted_index < len(classes_list):
                st.markdown(f'The predicted outcome is: `{classes_list[predicted_index]}`')
            else:
                st.error("The prediction is out of range. Please check your model output.")


    # Specify the columns to display
    columns_to_display = ["Gender", 
                          "Age", 
                          "Occupation", 
                          "Sleep Duration", 
                          "Quality of Sleep", 
                          "Physical Activity Level", 
                          "Stress Level", 
                          "BMI Category", 
                          "Blood Pressure", 
                          "Heart Rate", 
                          "Daily Steps",
                          "Sleep Disorder"]

    # Create 3 Data Frames containing 5 rows for each Sleep Disorder
    none_samples = sleep_df3[sleep_df3["Sleep Disorder"] == "None"].head(5)[columns_to_display]
    insomnia_samples = sleep_df3[sleep_df3["Sleep Disorder"] == "Insomnia"].head(5)[columns_to_display]
    sleepApnea_samples = sleep_df3[sleep_df3["Sleep Disorder"] == "Sleep Apnea"].head(5)[columns_to_display]

    if show_dataset_supervised:
        st.subheader("Dataset")
        st.dataframe(sleep_df3[columns_to_display], use_container_width=True, hide_index=True)

    if show_classes_supervised:
        # None Samples
        st.subheader("None Samples")
        st.dataframe(none_samples, use_container_width=True, hide_index=True)

        # Insomnia Samples
        st.subheader("Insomnia Samples")
        st.dataframe(insomnia_samples, use_container_width=True, hide_index=True)

        # Sleep Apnea Samples
        st.subheader("Sleep Apnea Samples")
        st.dataframe(sleepApnea_samples, use_container_width=True, hide_index=True)

    if show_none:
        # Display the None samples 
        st.subheader("None Samples")
        st.dataframe(none_samples, use_container_width=True, hide_index=True)

    if show_insomnia:
        # Display the Insomnia samples
        st.subheader("Insomnia Samples")
        st.dataframe(insomnia_samples, use_container_width=True, hide_index=True)

    if show_sleepApnea:
        # Display the Sleep Apnea samples
        st.subheader("Sleep Apnea Samples")
        st.dataframe(sleepApnea_samples, use_container_width=True, hide_index=True)





# Conclusions page

elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion:")
    st.markdown("""
    Sleep is an important factor in our lives as it helps us recharge and gain energy for the day. Typically, it is recommended for adults to get at least 8 hours of sleep each day, though not everyone can achieve this. There are different factors that can affect an individual's ability to get proper sleep. We will analyze the "Sleep Health and Lifestyle Dataset" by Tharamalingam, Laksika to better understand these factors. This dataset provides information on sleep, lifestyle, and cardiovascular health.

    The goal of this project is to...
    1) Analyze or label if a person gets "Good", "Middle", or "Bad" quality of sleep based on demographic, lifestyle, and cardiovascular health factors (unsupervised model)
    2) Predict which sleep disorder a person has based on certain factors (supervised model)

    Through data analysis and training of 3 unsupervised and 2 supervised models on the Sleep Health and lifestyle dataset, the key insights and observations are:

    1. Dataset Characteristics

    - The dataset shows the difference in sleep quality/sleep disorder depending on different factors that affect the human body. This can matter from what job they have to even what body weight they have. 

    2. Feature Distributions and Separability

    - Based on the different scatter plots, we noticed something specific in regards to the plots regarding Sleep Disorders in regards to Age as well as Sleep Disorders in regards to occupation. 

    - Compared to the different scatter plots, you can see that nearly all of them are balanced except for the ones in regards to age and occupation. 

    - Some occupation have a more common effect to sleep disorders than others like being a software engineer or a scientist and in regards to age, sleep apnea tend to be present at an early to late age while insomnia is more on the younger side of the spectrum. Besides that, there are still some people who have no sleep disorders regardless of their age. 


    3. Model Performance (Decision Tree Classifier)
    In regards to the Model Performance of the Decision Tree Classifier, we got an accuracy of 71.43%. 
    The top 3 features of importance are BloodPressure_Num (0.526885), BMI Category_Num (0.342760), and 
    Sleep Duration (0.092312). The enhanced supervised model got an accuracy of 75.1%.
    Roughly 3.33 increase in accuracy compared to the original Decision Tree Classifier.

    4. Model Performance (Random Forest Regressor)
    In regards to the Model Performance of the Random Forest Regressor, it got a train accuracy of 87.56% 
    and test accuracy of 19.84%. The calculated feature importance is BMI category (0.247572%) 
    and Blood Pressure  (0.752428 %). From this information, BMI and Blood Pressure is a big factor
    when identifying a sleep disorder. Random Forest Regressor is a better model when identifying sleep
    disorder.

    Summing up:

    Throughout this data science activity, it is evident that the data set "Sleep Health and Lifestyle" is a good dataset to use for classification and prediction. Due to its somewhat balanced yet specific data, it is actually a good data set to use. There were some cleaning up to do in data cleansing but afterwards it was okay to be used. As we can all tell, this data set is not really a precise future visioning of possibilities in regards to Sleep Health and Lifestyle but it can be more of a guide to those who are worried about their sleep.
""")