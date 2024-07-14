import streamlit as st
import pandas as pd
st.set_page_config(page_icon=":bulb:",page_title="Customer Segmentation and Predictive Analysis for B2C Marketplaces",initial_sidebar_state="collapsed")

st.title("Welcome:- Customer Segmentation and Predictive Analysis for B2C Marketplaces")
st.title("LOGIN ")
import re
# To hide side bar we use :-
st.markdown(
    """
        <style>
    [data-testid="stSidebar"]{   
        visibility: hidden;
    }
    [data-testid="stSidebarNavLink"]{
        visibility: hidden;
    }

    .reportview-container {
        margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
        footer {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none
    }
    </style>
    """,
    unsafe_allow_html=True
)
df = pd.read_csv("data.csv")
data = pd.DataFrame(df)

# Get the user input
email = st.text_input("Enter email")
password = st.text_input("Enter Password",type="password")

# Check if the user clicked the login button
try:
    if st.button("Login"):
        if data[data["Email"] == email]["Password"].values[0] == password:
            st.success("Login successful")
            st.markdown(f'<meta http-equiv="refresh" content="2;url=http://localhost:8501/main">', unsafe_allow_html=True)
            st.header("Redirecting...")
        else:
            st.error("Invalid Email Or Password")
except:
    st.warning("Enter email And Password")


link='[Register](http://localhost:8501/Signup)'
st.markdown(link,unsafe_allow_html=True)
