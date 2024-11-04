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
    2. Kobe Lituañas
    3. Zeus Jebril A. Kubaron
    4. John Augustine Caluag
    5. Joaquin Xavier Lajom
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
    if 'data' not in st.session_state:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Dataset Loaded Successfully!")
            st.write("### Dataset Preview:")
            st.dataframe(st.session_state.data.head())
    else:
        st.write("### Dataset Preview:")
        st.dataframe(st.session_state.data.head())

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

# Data Cleaning / Pre-processing Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning and Data Pre-processing")

    if 'data' in st.session_state:
        data = st.session_state.data.copy()

        # Display the original dataset for reference
        st.subheader("Original Dataset:")
        st.write(data)

        # Data Cleaning Options
        st.subheader("Data Cleaning Options")

        if st.button("Remove Null Values"):
            data = data.dropna()
            st.session_state.data = data
            st.success("Null values removed successfully!")
            st.write("### Data after removing null values:")
            st.dataframe(data.head())

        if st.button("Remove Duplicates"):
            data = data.drop_duplicates()
            st.session_state.data = data
            st.success("Duplicates removed successfully!")
            st.write("### Data after removing duplicates:")
            st.dataframe(data.head())

        # Dropping unused column 'Person ID'
        if st.button("Drop 'Person ID' Column"):
            if 'Person ID' in data.columns:
                data = data.drop('Person ID', axis=1)
                st.session_state.data = data
                st.success("Dropped 'Person ID' column successfully!")
                st.write("### Data after dropping 'Person ID':")
                st.dataframe(data.head())
            else:
                st.warning("The 'Person ID' column does not exist in the dataset.")

        # Summary Statistics
        if st.button("Show Summary Statistics"):
            st.subheader("Summary Statistics:")
            st.write(data.describe())

        # Encoding Categorical Variables
        st.subheader("Encoding Categorical Variables")
        categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
        if st.checkbox("Show Categorical Columns"):
            st.write("### Categorical Columns:")
            st.write(categorical_columns)

        if st.button("Encode Categorical Variables"):
            label_encoders = {}
            for col in categorical_columns:
                if data[col].dtype == 'object':
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    label_encoders[col] = le
            st.session_state.data = data
            st.success("Categorical variables encoded successfully!")
            st.write("### Encoded Data Preview:")
            st.dataframe(data.head())

        # Train-Test Split Section
        st.subheader("Train-Test Split")

        if 'data' in st.session_state:
            data = st.session_state.data

            # Ensure that target variable is selected and encoded
            target = 'Sleep Disorder'  # Set your target variable here

            if target not in data.columns:
                st.error(f"Target column '{target}' not found in the dataset.")
            else:
                if st.button("Perform Train-Test Split"):
                    X = data.drop(columns=[target])
                    y = data[target]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=y
                    )

                    # Store in session_state for access in other pages
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test

                    st.success("Train-Test split performed successfully!")

                    # Display Train-Test Split Details
                    st.write("### Train-Test Split Overview")
                    st.write("Dataset has been split into training and testing sets with a 70-30 ratio.")

                    st.write("#### X_train (Training Features)")
                    st.dataframe(X_train.head())

                    st.write("#### X_test (Testing Features)")
                    st.dataframe(X_test.head())

                    st.write("#### y_train (Training Labels)")
                    st.dataframe(y_train.head())

                    st.write("#### y_test (Testing Labels)")
                    st.dataframe(y_test.head())
    else:
        st.warning("Please upload a dataset in the Dataset page.")

# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    if 'X_train' in st.session_state and 'X_test' in st.session_state and 'y_train' in st.session_state and 'y_test' in st.session_state:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

        st.subheader("Model Training and Evaluation")

        # Model Selection
        model_type = st.selectbox("Select Model", ["Decision Tree Classifier", "Random Forest Classifier"])

        if model_type == "Decision Tree Classifier":
            st.markdown("""
            ### Decision Tree Classifier
            The Decision Tree Classifier is a machine learning algorithm used for classification tasks. It splits the data into subsets based on feature values, creating a tree-like structure where each node represents a decision based on a feature.
            """)

            # Hyperparameters
            max_depth = st.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)

            if st.button("Train Decision Tree Classifier"):
                dt_classifier = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                dt_classifier.fit(X_train, y_train)
                st.success("Decision Tree Classifier trained successfully!")

                # Predictions
                y_pred = dt_classifier.predict(X_test)

                # Evaluation
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                st.write(f"**Accuracy:** {accuracy * 100:.2f}%")
                st.write("**Classification Report:**")
                st.dataframe(report_df)

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

                # Feature Importance
                st.write("**Feature Importance:**")
                feature_importances = pd.Series(dt_classifier.feature_importances_, index=X_train.columns)
                st.dataframe(feature_importances.sort_values(ascending=False))

                # Feature Importance Plot
                fig2, ax2 = plt.subplots()
                sns.barplot(x=feature_importances.sort_values(ascending=False), y=feature_importances.sort_values(ascending=False).index, ax=ax2)
                ax2.set_title('Feature Importances')
                st.pyplot(fig2)

        elif model_type == "Random Forest Classifier":
            st.markdown("""
            ### Random Forest Classifier
            The Random Forest Classifier is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.
            """)

            # Hyperparameters
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth_rf = st.slider("Max Depth", 1, 20, 5)

            if st.button("Train Random Forest Classifier"):
                rf_classifier = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth_rf,
                    random_state=42
                )
                rf_classifier.fit(X_train, y_train)
                st.success("Random Forest Classifier trained successfully!")

                # Predictions
                y_pred_rf = rf_classifier.predict(X_test)

                # Evaluation
                accuracy_rf = accuracy_score(y_test, y_pred_rf)
                report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
                report_rf_df = pd.DataFrame(report_rf).transpose()

                st.write(f"**Accuracy:** {accuracy_rf * 100:.2f}%")
                st.write("**Classification Report:**")
                st.dataframe(report_rf_df)

                # Confusion Matrix
                cm_rf = confusion_matrix(y_test, y_pred_rf)
                fig_rf, ax_rf = plt.subplots()
                sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax_rf)
                ax_rf.set_xlabel('Predicted')
                ax_rf.set_ylabel('Actual')
                ax_rf.set_title('Confusion Matrix')
                st.pyplot(fig_rf)

                # Feature Importance
                st.write("**Feature Importance:**")
                feature_importances_rf = pd.Series(rf_classifier.feature_importances_, index=X_train.columns)
                st.dataframe(feature_importances_rf.sort_values(ascending=False))

                # Feature Importance Plot
                fig2_rf, ax2_rf = plt.subplots()
                sns.barplot(x=feature_importances_rf.sort_values(ascending=False), y=feature_importances_rf.sort_values(ascending=False).index, ax=ax2_rf)
                ax2_rf.set_title('Feature Importances')
                st.pyplot(fig2_rf)
    else:
        st.warning("Please complete the Data Cleaning and Pre-processing steps first.")

# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")
    st.write("This page will handle predictions based on trained models.")

    if 'data' in st.session_state and 'X_train' in st.session_state:
        data = st.session_state.data

        # Select model type for prediction
        model_type = st.selectbox("Select Trained Model for Prediction", ["Decision Tree Classifier", "Random Forest Classifier"])

        # Load the trained model from session_state if you have saved it
        # For simplicity, retrain the model here (in practice, you might want to save and load models)
        target = 'Sleep Disorder'  # Ensure this matches your target variable

        if model_type == "Decision Tree Classifier":
            st.subheader("Decision Tree Classifier Prediction")

            # Hyperparameters should match training
            max_depth = st.slider("Max Depth", 1, 20, 5)
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2)

            if st.button("Train Decision Tree for Prediction"):
                dt_classifier = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
                dt_classifier.fit(st.session_state.X_train, st.session_state.y_train)
                st.success("Decision Tree Classifier trained successfully!")

                # Input features for prediction
                st.subheader("Input Features for Prediction")
                input_data = {}
                for feature in st.session_state.X_train.columns:
                    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                input_df = pd.DataFrame([input_data])

                if st.button("Make Prediction"):
                    prediction = dt_classifier.predict(input_df)
                    label_encoder = LabelEncoder()
                    # Assuming you have stored the label encoder, else decode accordingly
                    # For demonstration, assuming labels are encoded as 0,1,2
                    st.write(f"**Predicted Sleep Disorder:** {prediction[0]}")
        
        elif model_type == "Random Forest Classifier":
            st.subheader("Random Forest Classifier Prediction")

            # Hyperparameters should match training
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth_rf = st.slider("Max Depth", 1, 20, 5)

            if st.button("Train Random Forest for Prediction"):
                rf_classifier = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth_rf,
                    random_state=42
                )
                rf_classifier.fit(st.session_state.X_train, st.session_state.y_train)
                st.success("Random Forest Classifier trained successfully!")

                # Input features for prediction
                st.subheader("Input Features for Prediction")
                input_data = {}
                for feature in st.session_state.X_train.columns:
                    input_data[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

                input_df = pd.DataFrame([input_data])

                if st.button("Make Prediction"):
                    prediction_rf = rf_classifier.predict(input_df)
                    label_encoder = LabelEncoder()
                    # Assuming you have stored the label encoder, else decode accordingly
                    # For demonstration, assuming labels are encoded as 0,1,2
                    st.write(f"**Predicted Sleep Disorder:** {prediction_rf[0]}")
    else:
        st.warning("Please complete the Data Cleaning and Pre-processing steps first.")

# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")
    st.write("""
    Summarize the insights and outcomes of your analysis here. Discuss the performance of your machine learning models, the importance of different features in predicting sleep disorders, and any recommendations or future work based on your findings.
    """)
