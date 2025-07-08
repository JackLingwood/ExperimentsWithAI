# # Working with Streamlit
# # https://streamlit.io/ 
# > streamlit run filename.py
# > https://localhost:8501
# > streamlit hello

import streamlit as st

st.title("Basic Chatbot")
st.title("_This is :blue[a title] :speech_balloon:")
st.title("$E = mc^2$")

st.header("This is a header")

st.subheader("This is a subheader")

st.write("This is a simple chatbot interface.")

st.text("Type your message below and click 'Send' to interact with the chatbot.")
st.text("This is plain text with no formatting.")

st.markdown("This is **bold text** and *italic text* in Markdown format. \n This is a list item")

st.write("You can also use Streamlit's built-in components to create interactive elements.")

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

st.write(data)


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



with st.chat_message("ai"): # user|assistant|human|ai
    st.write("Hello, I am a chatbot. How can I help you today?")

prompt = st.text_input("Type your message here:", placeholder="Enter your message...")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    
    # Simulate a response from the chatbot
    with st.chat_message("ai"):
        st.write(f"You said: {prompt}. This is a simulated response.")
    
    # Optionally, you can clear the input field after sending the message
    st.text_input("Type your message here:", placeholder="Enter your message...", key="new_input")






#st.write_stream("")
#st.chat_message("Hello, how can I assist you today?", is_user=False)


