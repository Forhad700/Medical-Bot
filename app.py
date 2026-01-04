import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Medical Bot Assistant", page_icon="⚕️")
st.title("🩺👩🏻‍⚕️ Medical Bot Assistant")

REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except:
    st.error("Missing HF_TOKEN in Streamlit Secrets!")
    st.stop()

client = InferenceClient(model=REPO_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """You are a professional Medical Triage Assistant. 
Analyze the symptoms provided and provide a possible triage priority (Emergency, Urgent, or Non-Urgent).
Always advise the user to seek professional medical help and provide clear 'Red Flag' warnings."""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("🩸 Describe Your Symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        try:
            # Added a check to ensure stream is actually starting
            stream = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5,
                stream=True 
            )
            
            for chunk in stream:
                # Only try to add content if it exists
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            
            # This is the important part: clear the cursor and save
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            # Only show error if we didn't get any response at all
            if not full_response:
                st.error("The server is busy. Please try sending your message again in 10 seconds.")
            else:
                # If we have a response, just finish it normally even if the connection cut at the very end
                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
