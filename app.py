import streamlit as st
import pandas as pd
import numpy as np

# Main title
st.title("Group Project Proposal: Streamlit Web App")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Cleaning", "Machine Learning"])

# Home Page
if page == "Home":
    st.header("Welcome to our Group Project Proposal Web App")
    st.write("Use the sidebar to navigate to different sections of the app.")

# Data Cleaning Page
elif page == "Data Cleaning":
    st.header("Data Cleaning Page")
    # Further data cleaning code will go here

# Machine Learning Page
elif page == "Machine Learning":
    st.header("Machine Learning Page")
    # Further ML code will go here
