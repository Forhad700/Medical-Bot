import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Medical Assistant", page_icon="⚕️")
st.title("🩺👩🏻‍⚕️ Medical Bot Assistant")

# Use Zephyr-7B: It is faster and has more free space than Llama or Mistral
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# Secure Token from Streamlit Secrets
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("Missing HF_TOKEN in Streamlit Secrets!")
    st.stop()

client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """You are a professional Medical Triage Assistant. 
Analyze the symptoms provided and provide a possible triage priority (Emergency, Urgent, or Non-Urgent).
Always advise the user to seek professional medical help."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("🩸 Describe Your Symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            # STREAMING VERSION: This shows the answer word-by-word
            stream = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5,
                stream=True 
            )
            
            placeholder = st.empty() # A spot to update the text live
            full_response = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌") # Adds a typing cursor
            
            placeholder.markdown(full_response) # Final text without cursor
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error("The server is busy. Please wait 10 seconds and try again.")
            st.info("Tip: If this keeps happening, Hugging Face free servers are full. Try again in a few minutes.")
