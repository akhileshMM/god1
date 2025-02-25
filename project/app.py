import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv
import os
import requests
# -------------------------------
# Step 1. PDF Processing Functions
# -------------------------------

@st.cache_data(show_spinner=False)
def load_pdf(pdf_path):
    pdf_pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pdf_pages.append(text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return pdf_pages

@st.cache_resource(show_spinner=False)
def create_vectorizer(pages):
    vectorizer = TfidfVectorizer().fit(pages)
    page_vectors = vectorizer.transform(pages)
    return vectorizer, page_vectors

def retrieve_context(query, vectorizer, page_vectors, pages, top_n=3):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, page_vectors).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    context = "\n\n-----\n\n".join([pages[i] for i in top_indices])
    return context

# -------------------------------
# Step 2. Initialize Groq API Client
# -------------------------------

client = Groq(api_key="")

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
