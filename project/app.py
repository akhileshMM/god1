import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
import os
import requests
from PyPDF2 import PdfReader
# -------------------------------
# Step 1. PDF Processing Functions
# -------------------------------

pages = []

try:
    reader = PdfReader("Bhagavad-GitaAsItis.pdf")  # Ensure this file exists
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]
except Exception as e:
    st.error(f"Error reading PDF: {e}")
    st.stop()

# Check if pages is still empty
if not pages:
    st.error("No text extracted from the PDF. Please check the file path.")
    st.stop()

# Debugging (optional)
st.write("Pages extracted:", pages)


# -------------------------------
# Step 2. Initialize Groq API Client
# -------------------------------
api_key = st.secrets["groq"]["api_key"]
client = Groq(api_key=api_key)

# -------------------------------
# Step 3. Streamlit Interface
# -------------------------------

st.sidebar.markdown("[Give Feedback Here!](https://forms.gle/your-google-form-link)")
pdf_url = "https://drive.google.com/file/d/1ezrjWqiOq9tFQrDycU7QfZcpDLO9kBz4/view?usp=sharing"
response = requests.get(pdf_url)
with open("Bhagavad-GitaAsItis", "wb") as f:
    f.write(response.content)

if not pages:
    st.error("No text extracted from the PDF. Please check the file path or PDF content.")
    st.stop()

vectorizer, page_vectors = create_vectorizer(pages)

st.subheader("Chat with the Bhagavad Gita")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def stream_gita_response(user_input, context):
    # Create a system message that includes the PDF-based context.
    system_message = (
        "You are Bhagavad Gita, the divine scripture. Answer the following question strictly based on the text provided below. "
        "Do not add any external knowledge or personal opinions. Use only the given text as your source.\n\n"
        "Context from Bhagavad Gita:\n"
        "-------------------------\n"
        f"{context}\n"
        "-------------------------\n"
        "Now, answer the following question:"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_input}
    ]
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=True,
        )
        response_text = ""
        for chunk in completion:
            content = getattr(chunk.choices[0].delta, "content", "")
            if content:
                response_text += content
                yield response_text  # Update the UI with the streaming response
    except Exception as e:
        yield f"❌ Error: {e}"

# Use chat_input so conversation history is maintained
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.spinner("Fetching response..."):
        # Retrieve context from the PDF based on the query
        context = retrieve_context(user_input, vectorizer, page_vectors, pages, top_n=3)
        full_response = ""
        for chunk in stream_gita_response(user_input, context):
            full_response = chunk
    st.session_state.chat_history.append(("Bhagavad Gita", full_response))

# Display the chat history
for speaker, message in st.session_state.chat_history:
    st.write(f"**{speaker}:** {message}")

st.markdown("---")
st.caption("Built as a demo to simulate a product experience. Your feedback is appreciated!")
