# # Working with Streamlit
# # https://streamlit.io/ 
# #  
# # ---------------------------------------------------------------------------------------------
# from dotenv import load_dotenv, find_dotenv
# import os
# import sys

# # Add Shared folder to sys.path
# sys.path.append(os.path.abspath("Shared"))

# from utils import heading, heading2, heading3, clearConsole, note, print_ascii_table , highlight_differences_diff_based

# from openai import OpenAI

# # Load environment variables
# load_dotenv(find_dotenv(), override = True)

# # Clear console and set working directory
# clearConsole()

# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
# print("Current working directory:", os.getcwd())

# # Headings
# name = os.path.basename(__file__)
# heading(f"{name}")

# # Environment variables
# api_key = os.environ.get("api_key")
# pinecone_vector_database_key = os.environ.get("pinecone_vector_database_key")
# pinecone_environment = os.environ.get("pinecone_environment", "gcp-starter")
# # ---------------------------------------------------------------------------------------------


import streamlit as st
st.title("Streamlit Nested Buttons Example")
# streamlit run c:/Code/ExperimentsWithAI/ChatBot/ChatBot - 1 - Streamlit.py
# https://localhost:8501


exit()

st.title("Nested Buttons Example")

if 'show_second_button' not in st.session_state:
    st.session_state.show_second_button = False
if 'second_button_clicked' not in st.session_state:
    st.session_state.second_button_clicked = False

# First button
if st.button("First Button"):
    st.session_state.show_second_button = True

# Check the state of the first button
if st.session_state.show_second_button:
    st.write("Revealed")

    # Second button
    if st.button("Second Button"):
        st.session_state.second_button_clicked = True

# Check the state of the second button
if st.session_state.second_button_clicked:
    st.write("Second Button Clicked!")
