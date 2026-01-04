import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Medical Assistant", page_icon="⚕️")
st.title("🩺👩🏻‍⚕️ Medical Bot Assistant")

# POINT TO A STABLE BASE MODEL INSTEAD
# This model is almost always 'warm' and ready on HF servers
REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct" 

# Secure Token from Streamlit Secrets
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("Missing HF_TOKEN in Streamlit Secrets!")
    st.stop()

client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

# MEDICAL SYSTEM PROMPT: This is the 'Brain' of your bot
# This tells the base Llama model to act exactly like your trained medical bot
SYSTEM_PROMPT = """You are a professional Medical Triage Assistant. 
Analyze the symptoms provided and provide a possible triage priority (Emergency, Urgent, or Non-Urgent).
Always advise the user to seek professional medical help."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("🩸 Describe Your Symptoms Here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            # We use the Chat Completion API which is more stable
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.5
            )
            answer = response.choices[0].message.content
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error("The server is currently busy. Please wait 10 seconds and try again.")
            st.info("Technical Hint: If this persists, the Llama-3 API might be at its free-tier limit.")
