import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Medical AI", page_icon="⚕️")
st.title("⚕️ Medical Assistant")

# Your merged model name
REPO_ID = "aay2y/medical-triage-bot"

# This looks for the token in Streamlit's hidden settings
# instead of having it typed out in the code
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("Missing HF_TOKEN in Streamlit Secrets!")
    st.stop()

client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            # Using the chat_completion method to talk to your model
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250
            )
            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:

            st.error("The model is waking up. Please wait 30 seconds and try again.")

