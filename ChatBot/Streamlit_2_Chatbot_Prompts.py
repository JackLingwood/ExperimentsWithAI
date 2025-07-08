import streamlit as st

with st.chat_message("ai"): # user|assistant|human|ai
    st.write("Hello, I am a chatbot. How can I help you today?")

prompt = st.text_input("Type your message here:", placeholder="Enter your message...", max_chars=500, key="user_input")
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
