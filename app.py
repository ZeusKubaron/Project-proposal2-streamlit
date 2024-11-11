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
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

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
    st.markdown("""
    1. Julianna Chanel Boado
    2. John Augustine Caluag
    3. Zeus Jebril A. Kubaron
    4. Joaquin Xavier S. Lajom
    5. Kobe Lituañas
    """)

#######################
# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")
    st.write("""
    Welcome to our **Sleep Health and Lifestyle Analysis Dashboard**. This dashboard showcases our project proposal, including data cleaning, exploratory data analysis (EDA), machine learning models, predictions, and conclusions.
    """)

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.write("Upload and explore your dataset here.")

    # Only upload if it's not already in session state
    if "data" not in st.session_state:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Dataset Loaded Successfully!")

    if "data" in st.session_state:
        data = st.session_state.data

        st.markdown("""
                        ### Content
                        - **Comprehensive Sleep Metrics**: Explore sleep duration, quality, and factors influencing sleep patterns.
                        - **Lifestyle Factors**: Analyze physical activity levels, stress levels, and BMI categories.
                        - **Cardiovascular Health**: Examine blood pressure and heart rate measurements.
                        - **Sleep Disorder Analysis**: Identify the occurrence of sleep disorders such as Insomnia and Sleep Apnea.\n\n """)

        st.write("### Dataset displayed as Data Frame:")
        st.write(
            "This dataset contains 400 rows and 13 columns related to sleep health and lifestyle factors.\n"
        )
        st.dataframe(data, use_container_width=True, hide_index=True)

        st.write("""\n
            ### Details about Sleep Disorder Column:

            - **None**: The individual does not exhibit any specific sleep disorder.
            - **Insomnia**: The individual experiences difficulty falling asleep or staying asleep, leading to inadequate or poor-quality sleep.
            - **Sleep Apnea**: The individual suffers from pauses in breathing during sleep, resulting in disrupted sleep patterns and potential health risks.""")

        # Describe Descriptive Statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe(), use_container_width=True)

        st.markdown("### Key Observations")
        st.write("""
            - **Sleep Duration**: The average sleep duration is around 7 hours, aligning with standard sleep recommendations for adults.
            - **Quality of Sleep**: The mean sleep quality score is about 6.5, suggesting moderate overall sleep quality.
            - **Stress Levels**: The dataset reveals an average stress level of around 5.8, indicating moderate stress among participants.
            - **Physical Activity**: With a mean score of approximately 6.2, the data reflects a moderately active population sample.
            - **Daily Steps**: The average daily step count is around 8000, slightly below the commonly recommended 10,000 steps per day.
            - **Blood Pressure**: The mean blood pressure is approximately 120 mmHg, typical for a healthy adult population.
            - **Alcohol and Caffeine Consumption**: Both metrics show moderate average consumption levels, around 3.5 and 4 respectively. """)

        st.write("""
                This overview provides a comprehensive summary of the dataset, detailing key aspects of sleep health
                and lifestyle factors. With this information, you can proceed to deeper analyses or modeling to explore
                relationships between these variables, such as the impact of stress and physical activity on sleep quality. """)

    else:
        st.write("No dataset loaded. Please upload a CSV file to proceed.")

# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    if 'data' in st.session_state:
        data = st.session_state.data

        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(data.describe())

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        corr = data.select_dtypes(include=['float64', 'int64']).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(plt)

        # Pairplot
        st.subheader("Pairplot")
        pairplot_fig = sns.pairplot(data.select_dtypes(include=['float64', 'int64']))
        st.pyplot(pairplot_fig)
    else:
        st.warning("Please upload a dataset in the Dataset page.")

##Data Cleaning

###Checking the data frame


sleep_df.duplicated()

"""There is no duplicated data"""

sleep_df.info()

"""In the sleep disorder column there is 155 non-null values/data"""

# Categorical columns
cat_col = [col for col in sleep_df.columns if sleep_df[col].dtype == 'object']
print('Categorical columns :',cat_col)

# Numerical columns
num_col = [col for col in sleep_df.columns if sleep_df[col].dtype != 'object']
print('Numerical columns :',num_col)

"""The categorical coulumns are Gender, Occupation, BMI Category, Blood Pressure, and Sleep Disorder. On the other hand, the numerical columns are Person ID, Age, Sleep Duration, Quality of Sleep, Physical Activity Level, Stress Level, Heart Rate, and Daily Steps."""

sleep_df[cat_col].nunique()

"""It displays number values/data seen in the categorical columns"""

sleep_df['Blood Pressure'].unique()

"""The blood pressure column is consider categorical because of the '/'. Here it displays the different blood pressure seen the data frame /dataset"""

round((sleep_df.isnull().sum()/sleep_df.shape[0])*100,2)

"""It shows that the Sleep Disorder column caontains a total of 58.56% null values"""

sleep_df.isnull().sum()

"""There are 219 entries of null values in the sleep disorder column"""

sleep_df[['Sleep Disorder']].value_counts()

"""**NOTE!!!**

Not all people have a sleep disorder which is shown in our chosen dataset. On the Sleep Disorder column, there are three values which are "None", "Insomnia", and "Sleep Apnea". The program consider the word "NONE" as a null value in the Sleep Disorder column.

Code below will convert/replace "NaN" into "None" so that it will display the 3 given values.

###Fixing the null values (convert/replace "NaN" into "None")
"""

# Create a new DataFrame
sleep_df2 = sleep_df.copy()

# Replace NaN values with "None" in the 'Sleep Disorder' column
sleep_df2['Sleep Disorder'] = sleep_df2['Sleep Disorder'].fillna("None")

sleep_df2.head()

sleep_df2[['Sleep Disorder']].value_counts()

"""In the sleep_df2 dataframe, there are now 3 values and the sleep disorder column will not have any null values"""

# Pie chart for the sleep disorder column
def pie_chart_summary():
    disorder_counts = sleep_df2['Sleep Disorder'].value_counts()
    labels = disorder_counts.index

    # Plot pie chart
    plt.pie(disorder_counts, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart of Sleep Disorder')
    plt.show()

pie_chart_summary()

"""There is a new problem in the dataset. The 3 values under sleep disorder are not equally distributed. 58.6% people in the dataset has "None" or does not have sleep disorder. 20.6% has insomnia while 20.9% have Sleep Apnea. The three values needs to be equal to ensure them machine learning models will correctly predict which sleep disorder a person has.

---

**The codes below will balance out the data**

---

###Make the data balance
"""

#Count the instances of each value in sleep disorder (label)
label_counts = sleep_df2[['Sleep Disorder']].value_counts()

#Find the minimum instances among the three values to use as the target count
min_count = label_counts.min()

#Sample each category to match the minimum count
sleep_df3 = (
    sleep_df2
    .groupby('Sleep Disorder', as_index=False)
    .apply(lambda x: x.sample(min_count))
    .reset_index(drop=True)
)

sleep_df3.head()

sleep_df3['Sleep Disorder'].value_counts()

# Pie chart for the sleep disorder column
def pie_chart_summary():
    disorder_counts = sleep_df3['Sleep Disorder'].value_counts()
    labels = disorder_counts.index

    # Plot pie chart
    plt.pie(disorder_counts, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart of Sleep Disorder')
    plt.show()

pie_chart_summary()

"""The dataset is now balance in the 'sleep_df3' data frame. All 3 values in the sleep disorder column are equally 33.3%

###Categorical to Numerical
"""

encoder = LabelEncoder()

sleep_df3['Gender_Num'] = encoder.fit_transform(sleep_df3['Gender'])

sleep_df3['Occupation_Num'] = encoder.fit_transform(sleep_df3['Occupation'])

sleep_df3['BMICategory_Num'] = encoder.fit_transform(sleep_df3['BMI Category'])

sleep_df3['BloodPressure_Num'] = encoder.fit_transform(sleep_df3['Blood Pressure'])

sleep_df3['SleepDisorder_Num'] = encoder.fit_transform(sleep_df3['Sleep Disorder'])

"""Transform the categorical values into numerical using label encoder

'Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder'
"""

# Mapping of the Gender and their encoded equivalent

categorical_col = sleep_df3['Gender'].unique()
encoded_col = sleep_df3['Gender_Num'].unique()

# Create a new DataFrame
gender_mapping_df = pd.DataFrame({'Gender': categorical_col, 'Gender_Num': encoded_col})

# Display the DataFrame
gender_mapping_df

"""Displays the encoded equivalent of Gender"""

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

"""Displays the encoded equivalent of BMI"""

# Mapping of the BP and their encoded equivalent

categorical_col = sleep_df3['Blood Pressure'].unique()
encoded_col = sleep_df3['BloodPressure_Num'].unique()

# Create a new DataFrame
bp_mapping_df = pd.DataFrame({'Blood Pressure': categorical_col, 'BloodPressure_Num': encoded_col})

# Display the DataFrame
bp_mapping_df

"""Displays the encoded equivalent of Blood Pressure"""

# Mapping of the Sleep Disorder and their encoded equivalent

categorical_col = sleep_df3['Sleep Disorder'].unique()
encoded_col = sleep_df3['SleepDisorder_Num'].unique()

# Create a new DataFrame
sleepdisorder_mapping_df = pd.DataFrame({'Sleep Disorder': categorical_col, 'SleepDisorder_Num': encoded_col})

# Display the DataFrame
sleepdisorder_mapping_df

"""Displays the encoded equivalent of Sleep Disorder"""

sleep_df3.head()

"""Show the data frame with the encoded equivalent of the categorical columns

###Dropping columns that will not be used
"""

sleep_df3.drop('Person ID', axis=1, inplace=True)

"""The Person Id column is remove because it does not help predicting what sleep disoder a person has and doen not help detrimine th quality of the sleep a person has



"""##Machine Learning Implementations

###Unsupervised
"""
Demographic factors
- Age
- Gender
- Occupation

Lifestyle factors
- Sleep Duration
- Physical Activity Level
- Stress Level
- Daily Steps

Cardiovascular health factors
- Heart Rate
- Blood Pressure:
- BMI Category
- Sleep Disorder

###Demographic Factors

"""
kmeans = KMeans(n_clusters=3, random_state=0)
sleep_df3['Cluster_demographic'] = kmeans.fit_predict(sleep_df3[['Quality of Sleep','Age', 'Gender_Num', 'Occupation_Num']])

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Quality of Sleep', 'Age', 'Gender_Num', 'Occupation_Num'])
print("Cluster Centroids:")
print(centroids)

"""Based on the calculated center value of each clusters, cluster 1 has the highest quality of sleep, followed by cluster 0, while cluster 2 has the lowest."""

#Create a mapping of cluster labels based on your new assignment
Cluster_demographic_labels = {
    0: "Moderate Sleep",  # Label for Cluster 0
    1: "Good Sleep",    # Label for Cluster 1
    2: "Bad Sleep"      # Label for Cluster 2
}

sleep_df3['Cluster_demographic_labels'] = sleep_df3['Cluster_demographic'].map(Cluster_demographic_labels)

sleep_df3[['Quality of Sleep', 'Age', 'Gender_Num', 'Occupation_Num', 'Cluster_demographic', 'Cluster_demographic_labels']]

sns.pairplot(sleep_df3[['Quality of Sleep', 'Age', 'Gender_Num', 'Occupation_Num', 'Cluster_demographic_labels']], hue='Cluster_demographic_labels', palette='viridis')
plt.show()

"""Cluster 1 represents good quality sleep, with the highest average score of 7.7. The people are aged about 53.5 years, mostly women (Gender_Num 0.05), and are probably teachers or similar (Occupation_Num 4.05). Cluster 0 has moderate sleep quality, averaging a score of 6.8. Its people are younger, around 33.9 years old, with a slight male majority (Gender_Num 0.76), and and most probably engineers (Occupation_Num 2.82). Cluster 2 has the lowest average score of 6.7 of quality of sleep, with an average age of 43.7 years, a balanced gender distribution (Gender_Num 0.59), and an Occupation_Num of 7.25 suggests many are software engineers.

####Lifestyle Factors
"""

kmeans = KMeans(n_clusters=3, random_state=0)
sleep_df3['Cluster_lifestyle'] = kmeans.fit_predict(sleep_df3[['Quality of Sleep','Sleep Duration', 'Physical Activity Level', 'Stress Level']])

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Quality of Sleep','Sleep Duration', 'Physical Activity Level', 'Stress Level'])
print("Cluster Centroids:")
print(centroids)

"""Based on the calculated center value of each clusters, cluster 0 has the highest quality of sleep, followed by cluster 2, while cluster 1 has the lowest."""

#Create a mapping of cluster labels based on your new assignment
Cluster_lifestyle_labels = {
    0: "Good Sleep",  # Label for Cluster 0
    1: "Bad Sleep",    # Label for Cluster 1
    2: "Moderate Sleep"      # Label for Cluster 2
}

sleep_df3['Cluster_lifestyle_labels'] = sleep_df3['Cluster_lifestyle'].map(Cluster_lifestyle_labels)

sleep_df3[['Quality of Sleep','Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Cluster_lifestyle', 'Cluster_lifestyle_labels']]

sns.pairplot(sleep_df3[['Quality of Sleep','Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Cluster_lifestyle_labels']], hue='Cluster_lifestyle_labels', palette='viridis')
plt.show()

"""Cluster 0 represents good quality sleep, with a score of 8.2. Individuals who average 7.67 hours of sleep have a high physical activity level of 69.05, while their stress level is at 4.22, moderate. Cluster 1 indicates mid-quality sleep, with a score of 6.5. This group averages 6.59 hours of sleep, has lower physical activity at 39.85, and a higher stress level of 6.08, suggesting that reduced activity and high stress may impact their sleep. Cluster 2 reflects poor sleep quality, scoring 6.7. Members also average 6.59 hours of sleep, but their high physical activity level of 89.77 and elevated stress level of 7.00 indicate that stress may be a key factor affecting sleep quality.

####Cardiovascular health factors
"""

kmeans = KMeans(n_clusters=3, random_state=0)
sleep_df3['Cluster_cardiovascularH'] = kmeans.fit_predict(sleep_df3[['Quality of Sleep','Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num']])

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Quality of Sleep','Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num'])
print("Cluster Centroids:")
print(centroids)

"""Based on the calculated center value of each clusters, cluster 0 has the highest quality of sleep, followed by cluster 1, while cluster 2 has the lowest."""

#Create a mapping of cluster labels based on your new assignment
Cluster_cardiovascularH_labels = {
    0: "Good Sleep",  # Label for Cluster 0
    1: "Moderate Sleep",    # Label for Cluster 1
    2: "Bad Sleep"      # Label for Cluster 2
}

sleep_df3['Cluster_cardiovascularH_labels'] = sleep_df3['Cluster_cardiovascularH'].map(Cluster_cardiovascularH_labels)

sleep_df3[['Quality of Sleep','Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num', 'Cluster_cardiovascularH', 'Cluster_cardiovascularH_labels']]

sns.pairplot(sleep_df3[['Quality of Sleep','Heart Rate', 'BloodPressure_Num', 'BMICategory_Num', 'SleepDisorder_Num', 'Cluster_cardiovascularH_labels']], hue='Cluster_cardiovascularH_labels', palette='viridis')
plt.show()

"""Cluster 0 represents good quality sleep, with an average score of 7.3. Participants have an average heart rate of 69.2 bpm, a BloodPressure_Num of 2.57 (around 132/87), a normal BMI (BMICategory_Num 0.17), and a low prevalence of sleep disorders (SleepDisorder_Num 1). Cluster 1 shows moderate sleep quality, with an average score of 7.1. This group has a slightly elevated heart rate of 74 bpm, a BloodPressure_Num of 16.06 (around 140/90), a tendency towards overweight (BMICategory_Num 2.79), and a moderate likelihood of sleep disorders (SleepDisorder_Num 1.76). Cluster 2 has the lowest quality of sleep, with an average score of 6.9. Participants have a heart rate of 69.3 bpm, a BloodPressure_Num of 10.3 (between 130/85 and 135/90), lean towards overweight (BMICategory_Num 2.14), and show signs of insomnia (SleepDisorder_Num 0.34), indicating poor sleep quality in this group.

###Supervised
"""

sleep_df3.head()

sleep_df3['Sleep Disorder'].unique()

"""Checking again all the unique values in the Sleep Disorder column"""

sleep_df3['Sleep Disorder'].value_counts()

"""All the entries in the Sleep Disorder column are already balanced"""

# Extract unique values for the Sleep Disorder columns
unique_sleepD = sleep_df3['Sleep Disorder'].unique()
unique_sleepD_num = sleep_df3['SleepDisorder_Num'].unique()

# Create a new DataFrame with the unique values side-by-side
sleepDisorder_unique_values_df = pd.DataFrame({
    'Unique Column1': unique_sleepD,
    'Unique Column2': pd.Series(unique_sleepD_num)
})

# Display the result
sleepDisorder_unique_values_df

# Select features and target variable
features = ['Gender_Num', 'Age', 'Occupation_Num', 'Sleep Duration', 'Quality of Sleep',
            'Physical Activity Level', 'Stress Level', 'BMICategory_Num', 'BloodPressure_Num',
            'Heart Rate', 'Daily Steps']
x = sleep_df3[features]
y = sleep_df3['SleepDisorder_Num']

X

y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape

X_train.head()

X_test.shape

X_test.head()

y_train.shape

y_train.head()

y_test.shape

y_test.head()

# Train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

"""####Model Evaluation"""

# Evaluate the model
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

feature_importance = dt_classifier.feature_importances_

feature_importance

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Calculate the importance as a percentage
importance_df['Importance (%)'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100

# Sort the DataFrame by importance for better readability
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display the resulting DataFrame
print(importance_df)

"""### Enhanced Supervised Model"""

# Select new features and new target variable
Newfeatures = ['BMICategory_Num', 'BloodPressure_Num']
NewX = sleep_df3[Newfeatures]
NewY = sleep_df3['SleepDisorder_Num']

NewX

NewY

# Split the dataset into training and testing sets
NewX_train, NewX_test, NewY_train, NewY_test = train_test_split(NewX, NewY, test_size=0.3, random_state=42)

NewX_train.shape

NewX_train.head()

NewX_test.shape

NewX_test.head()

NewY_train.shape

NewY_train.head()

NewY_test.shape

NewY_test.head()

# Train the Decision Tree Classifier
new_dt_classifier = DecisionTreeClassifier(random_state=42)
new_dt_classifier.fit(NewX_train, NewY_train)

"""####Model Evaluation

"""

# Evaluate the model
y_pred = new_dt_classifier.predict(NewX_test)
accuracy = accuracy_score(NewY_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')



# Prediction Page
# Conclusions Page

