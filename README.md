# 🩺 Medical Bot Assistant 💊

Developed a high-performance **Medical Bot Assistant** built with Fine-Tuned Llama-3 (8B) model to provide real-time symptom analysis and emergency prioritization. This project demonstrates the complete AI lifecycle: from Supervised Fine-Tuning (SFT) on medical datasets to deploying a Production-Ready web interface with real-time token streaming.

---
## Features
- **User Input** — User describes symptoms (e.g., "I have a sharp pain in my chest").
- **Analysis** — The model evaluates the severity and generates a structured response.
- **LLM Processing** — The Llama-3-8B model analyzes the input against its fine-tuned medical weights to determine clinical severity.
- **Intelligent Triage Engine** — Leverages a specialized medical persona to categorize patient symptoms into Emergency, Urgent, or Non-Urgent priority levels.
- **Low-Latency Streaming** — Implemented word-by-word "progressive rendering" to eliminate user-perceived wait times (TTFT).
- **Clinical Guardrails** — Automated "Red Flag" detection logic that triggers immediate emergency protocols for life-threatening scenarios.
- **Production Resiliency** — Integrated a Smart-Retry Mechanism and error-handling layer to maintain 99% service uptime on serverless infrastructure.
---

## Tech Stack
- **Model** — Llama-3-8B-Instruct (Domain-adapted for medical triage)
- **Fine-Tuning** — SFT (Supervised Fine-Tuning) using PEFT/QLoRA in Google Colab.
- **Language** — Python
- **Frontend** — Streamlit (Reactive Web UI)
- **Backend** — Hugging Face Inference API (Serverless LLM Hosting)
---
## Pipeline (How It Works)
- **Brain (Fine-Tuning)** — The base Llama-3 model was adapted via QLoRA (Quantized Low-Rank Adaptation) to learn medical vocabulary and triage logic while significantly reducing memory overhead.
- **Connection (Inference)** — The app communicates with a hosted version of the model via the Hugging Face Inference Client, using Direct API Calls for maximum speed rather than heavy orchestration wrappers.
- **Experience (Streaming)** — Tokens are streamed from Hugging Face to the Streamlit UI for a fast, responsive user experience. Utilizing Server-Sent Events (SSE), the backend streams tokens to the frontend as they are generated, reducing the initial "Time to First Token" from 5s down to <500ms.
- **System Prompting** — The model is initialized with a professional "Medical Triage" persona.
- **Safety Layer** — A rigorous System Prompt acts as the clinical boundary, ensuring the AI maintains a professional tone and consistently provides medical disclaimers.

---
## ⚠️ Disclaimer
This project is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.
