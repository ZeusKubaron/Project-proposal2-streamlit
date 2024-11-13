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

# rfr_classifier = joblib.load('assets/models/random_forest_regressor.joblib')
features = ["BMICategory_Num", "BloodPressure_Num"]
disorder_list = ["Insomnia", "None", "Sleep-Apnea	"]

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.write("""
    # About this Dashboard

    Welcome to the Sleep Health and Lifestyle Dashboard. This dashboard provides insights into how lifestyle factors affect sleep quality, along with predictive modeling based on machine learning algorithms. Below is a brief overview of each component:

    ---

    ### Dataset
    The **Sleep Health and Lifestyle Dataset** serves as the foundation of this analysis. This dataset captures various health and lifestyle indicators, helping us explore their impact on **Sleep Quality**.

    ---

    ### Exploratory Data Analysis (EDA)
    **EDA (Exploratory Data Analysis)** offers a comprehensive look into sleep quality and its relationship with lifestyle factors. Key visualizations include:
    - **Pie Charts** representing distribution of sleep health categories
    - **Scatter Plots** depicting relationships between lifestyle variables and sleep quality

    These visualizations help us understand patterns and trends in the data.

    ---

    ### Data Cleaning and Pre-processing
    Our data cleaning and pre-processing steps include essential tasks such as:
    - Encoding categorical variables (e.g., sleep quality categories)
    - Splitting the dataset into **training and testing sets**

    These steps ensure that our data is ready for effective model training and evaluation.

    ---

    ### Machine Learning Models
    We employed a mix of **unsupervised and supervised models** to analyze and predict sleep quality:
    - **3 Unsupervised Models** for clustering individuals based on lifestyle and health factors
    - **2 Supervised Models** to classify sleep quality based on health and lifestyle inputs

    ---

    ### Prediction
    On the **Prediction Page**, users can input their own values for lifestyle and health indicators. Based on these inputs, our trained models will provide predictions for sleep health. This feature allows users to explore personalized insights.

    ---

    ### Conclusion
    This section summarizes key insights and observations from the **EDA** and **model training** phases. We also highlight findings and patterns discovered throughout the analysis process, offering a well-rounded perspective on how lifestyle impacts sleep quality.

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
            - **Alcohol and Caffeine Consumption**: Both metrics show moderate average consumption levels, around 3.5 and 4 respectively.

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
        st.title("Demographic Factors")

        sleep_df3 = st.session_state.get("sleep_df3")

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

        st.write(
            "Based on the calculated center value of each clusters, cluster 1 has the highest quality of sleep, followed by cluster 0, while cluster 2 has the lowest."
        )

        # Create a mapping of cluster labels based on your new assignment
        Cluster_demographic_labels = {
            0: "Moderate Sleep",  # Label for Cluster 0
            1: "Good Sleep",  # Label for Cluster 1
            2: "Bad Sleep",  # Label for Cluster 2
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
            "Cluster 1 represents good quality sleep, with the highest average score of 7.7.  The people are aged about 53.5 years, mostly women (Gender_Num 0.05), and are probably teachers or similar (Occupation_Num 4.05). Cluster 0 has moderate sleep quality, averaging a score of 6.8. Its people are younger, around 33.9 years old, with a slight male majority (Gender_Num 0.76), and and most probably engineers (Occupation_Num 2.82). Cluster 2 has the lowest average score of 6.7 of quality of sleep, with an average age of 43.7 years, a balanced gender distribution (Gender_Num 0.59), and an Occupation_Num of 7.25 suggests many are software engineers."
        )

        st.title("Lifestyle Factors")

        kmeans = KMeans(n_clusters=3, random_state=0)
        sleep_df3["Cluster_lifestyle"] = kmeans.fit_predict(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Sleep Duration",
                    "Physical Activity Level",
                    "Stress Level",
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
            ],
        )
        st.write("Lifestyle Cluster Centroids:")
        st.write(centroids)

        # Create a mapping for the lifestyle cluster labels (same as original)
        Cluster_lifestyle_labels = {
            0: "Good Sleep",  # Label for Cluster 0
            1: "Bad Sleep",  # Label for Cluster 1
            2: "Moderate Sleep",  # Label for Cluster 2
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
                    "Cluster_lifestyle_labels",
                ]
            ],
            hue="Cluster_lifestyle_labels",
            palette="viridis",
        )
        st.pyplot(plt)

        st.title("Cardiovascular Health Factors")

        kmeans = KMeans(n_clusters=3, random_state=0)
        sleep_df3["Cluster_cardiovascularH"] = kmeans.fit_predict(
            sleep_df3[
                [
                    "Quality of Sleep",
                    "Heart Rate",
                    "BloodPressure_Num",
                    "BMICategory_Num",
                    "SleepDisorder_Num",
                ]
            ]
        )

        centroids = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=[
                "Quality of Sleep",
                "Heart Rate",
                "BloodPressure_Num",
                "BMICategory_Num",
                "SleepDisorder_Num",
            ],
        )
        st.write("Cluster Centroids:")
        st.write(centroids)

        # Create a mapping of cluster labels based on your new assignment
        Cluster_cardiovascularH_labels = {
            0: "Good Sleep",  # Label for Cluster 0
            1: "Moderate Sleep",  # Label for Cluster 1
            2: "Bad Sleep",  # Label for Cluster 2
        }

        sleep_df3["Cluster_cardiovascularH_labels"] = sleep_df3[
            "Cluster_cardiovascularH"
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
            "Cluster 0 represents good quality sleep, with an average score of 7.3. Participants have an average heart rate of 69.2 bpm, a BloodPressure_Num of 2.57 (around 132/87), a normal BMI (BMICategory_Num 0.17), and a low prevalence of sleep disorders (SleepDisorder_Num 1). Cluster 1 shows moderate sleep quality, with an average score of 7.1. This group has a slightly elevated heart rate of 74 bpm, a BloodPressure_Num of 16.06 (around 140/90), a tendency towards overweight (BMICategory_Num 2.79), and a moderate likelihood of sleep disorders (SleepDisorder_Num 1.76). Cluster 2 has the lowest quality of sleep, with an average score of 6.9. Participants have a heart rate of 69.3 bpm, a BloodPressure_Num of 10.3 (between 130/85 and 135/90), lean towards overweight (BMICategory_Num 2.14), and show signs of insomnia (SleepDisorder_Num 0.34), indicating poor sleep quality in this group."
        )

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

# Prediction Page
elif st.session_state.page_selection == "prediction":
    if "sleep_df3" in st.session_state:
        sleep_df3 = st.session_state["sleep_df3"]

        #st.session_state['kmeans'] = kmeans  # Store the trained model in session state


        st.header("🔮 Prediction Page")

        
        # Unsupervised Part
        st.title("Unsupervised Models Prediction")
    
        col_pred1 = st.columns((3, 3, 3), gap="medium")

        with col_pred1[0]:
            st.markdown("#### 👥 Demographic Factors")

            # Input value for age under demographic factors
            dt_age = st.number_input('Age (27 - 60)', min_value=27, max_value=60, step=1, key='age')

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
                          'Sales Representative',
        ]

        # Mapping Gender and Occupation to Numerical Values
        gender_mapping = {"Male": 1, "Female": 0}
        occupation_mapping = {'Teacher': 9,
                              'Accountant':	0,
                              'Salesperson': 6,
                              'Nurse': 4, 
                              'Lawyer':	3,
                              'Doctor': 1,
                              'Engineer': 2,
                              'Software Engineer': 8,
                              'Scientist': 7,
                              'Sales Representative': 5}

        # Dropdown Menus
        dt_gender = st.selectbox("Gender", options=gender_values, index=0, key="dt_gender")
        dt_occupation = st.selectbox("Occupation", options=occupation_values, index=0, key="dt_occupation")

        # Class labels for prediction
        classes_list_demoFactor = ["Moderate Sleep", "Good Sleep", "Bad Sleep"]

        # Button to Detect Sleep Quality
        if st.button("Detect", key="dt_detect"):
            # Convert categorical inputs to numerical values
            dt_gender_num = gender_mapping[dt_gender]
            dt_occupation_num = occupation_mapping[dt_occupation]

            # Prepare the input data for prediction
            input_data = [[dt_age, dt_gender_num, dt_occupation_num]]

            # Load your CSV file
            #data = pd.read_csv("your_data.csv")  # Replace with the actual path to your CSV file

            # Assuming your data has columns 'age', 'gender', and 'occupation'
            # Ensure these columns are present in your dataset
            #if 'age' in data.columns and 'gender' in data.columns and 'occupation' in data.columns:
                # Scale the data
                #scaler = MinMaxScaler()
                #scaled_data = scaler.fit_transform(data[['age', 'gender', 'occupation']])

                # Train the KMeans model
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(scaled_data)

                # Scale the input data
            scaled_input_data = scaler.transform(input_data)

                # Predict the cluster using the trained KMeans model
            prediction = kmeans.predict(scaled_input_data)

                # Map the predicted cluster to sleep quality
            predicted_label = classes_list_demoFactor[prediction[0]]

                # Display the prediction result
            st.markdown(f"### The predicted sleep quality is: `{predicted_label}`")
        else:
            st.error("The required columns ('age', 'gender', 'occupation') are missing from the dataset.")

        with col_pred1[1]:
            st.markdown("#### 👥 Lifestyle Factors")

        with col_pred1[2]:
            st.markdown("#### 👥 Cardiovascular health Factors")







    
    #End of Unsupervised part
    ##################################################################################

    #Supervised part
    st.title("Supervised Models Prediction")

    col_pred2 = st.columns((1.5, 3, 3), gap='medium')

    # Initialize session state for clearing results
    if 'clear' not in st.session_state:
        st.session_state.clear = False

    with col_pred2[0]:
        with st.expander('Options', expanded=True):
            show_dataset = st.checkbox('Show Dataset')
            show_classes = st.checkbox('Show All Classes')
            show_none = st.checkbox('Show None')
            show_insomnia = st.checkbox('Show Insomnia')
            show_sleepApnea = st.checkbox('Show Sleep Apnea')

            clear_results = st.button('Clear Results', key='clear_results')

            if clear_results:

                st.session_state.clear = True
    


    with col_pred2[1]:
        st.markdown("####🌲 Decision Tree Classifier")
        
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
        if st.button('Detect', key='dt_detect'):
            # Map categorical inputs to numeric codes 
            gender_map = {'Male': 0, 'Female': 1}
            occupation_map = {'Teacher': 9,
                              'Accountant':	0,
                              'Salesperson': 6,
                              'Nurse': 4, 
                              'Lawyer':	3,
                              'Doctor': 1,
                              'Engineer': 2,
                              'Software Engineer': 8,
                              'Scientist': 7,
                              'Sales Representative': 5}
            bmi_map = {'Overweight', 
                       'Obese', 
                       'Normal'}
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
            Gender_Num = gender_map[gender]
            Occupation_Num = occupation_map[occupation]
            BMICategory_Num = bmi_map[bmi_category]
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
        st.markdown("####🌲🌲🌲 Random Forest Regressor")
        # empty for the part. just texting the divide thingy

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

    if show_dataset:
        st.subheader("Dataset")
        st.dataframe(sleep_df3[columns_to_display], use_container_width=True, hide_index=True)

    if show_classes:
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

    - The dataset shows the difference in sleep quality depending on different factors that affect the human body. This can matter from what job they have to even what body weight they have. 

    2. Feature Distributions and Separability

    - Based on the different scatter plots, we noticed something specific in regards to the plots regarding Sleep Disorders in regards to Age as well as Sleep Disorders in regards to occupation. 

    - Compared to the different scatter plots, you can see that nearly all of them are balanced except for the ones in regards to age and occupation. 

    - Some occupation have a more common effect to sleep disorders than others like being a software engineer or a scientist and in regards to age, sleep apnea tend to be present at an early to late age while insomnia is more on the younger side of the spectrum. Besides that, there are still some people who have no sleep disorders regardless of their age. 


    3. Model Performance ()

    4. Model Performance ()


    Summing up:

    Throughout this data science activity, it is evident that the data set "Sleep Health and Lifestyle" is a good dataset to use for classification and prediction. Due to its somewhat balanced yet specific data, it is actually a good data set to use. There were some cleaning up to do in data cleansing but afterwards it was okay to be used. As we can all tell, this data set is not really a precise future visioning of possibilities in regards to Sleep Health and Lifestyle but it can be more of a guide to those who are worried about their sleep.
""")