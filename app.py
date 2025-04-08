import os
import gradio as gr
import json

from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# === API Keys from Hugging Face Secrets ===
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# === Load JSON File ===
json_file_path = "merged_data.json"

loader = JSONLoader(file_path=json_file_path, jq_schema=".", text_content=False)
documents = loader.load()

# === Split Text ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(documents)

# === Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# === Pinecone Setup ===
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "ragshl"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

docsearch = PineconeVectorStore.from_documents(text_chunks, index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# === Prompt + LLM (Gemini) ===
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBuLDvD8NWtB3EDhQsal-ihmsCnckpjgt0"

# Load Gemini model (e.g., Gemini Pro)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash",temperature=0.4, max_tokens=500)

# Define your system prompt
system_prompt = (
    "You are a helpful and knowledgeable virtual assistant for the SHL website. "
    "Your job is to help users find the best SHL assessments from the product catalog based on their job role, level, industry, or other preferences.\n\n"

    "You have access to structured assessment data, where each test includes the following fields:\n"
    "- `name`: Name of the assessment\n"
    "- `url`: Link to the assessment on SHL's website\n"
    "- `description`: A short explanation of the test and what it evaluates\n"
    "- `test_type`: Type of the test (e.g., Knowledge-based \"K\", Behavioral, Personality)\n"
    "- `job_levels`: Suitable job levels (e.g., Entry, Mid-Professional, Leadership)\n"
    "- `languages`: Available languages for the assessment\n"
    "- `assesment_length`: Approximate duration of the assessment\n"
    "- `pdffile`: PDF overview or technical document (optional)\n\n"

    "Use this structured data to:\n"
    "- Recommend relevant assessments\n"
    "- Filter results based on user-specified criteria (e.g., “mid-level developers”, “available in Spanish”, “personality test for sales roles”)\n"
    "- Provide links and concise summaries from the catalog\n"
    "- Suggest alternative assessments if an exact match isn’t found\n"
    "- Always encourage users to refine or reset their search if needed\n\n"

    "If a user enters a keyword, try to match it against the `name`, `description`, or `job_levels`.\n\n"

    "Only provide results from the dataset. Never make up test names or data.\n\n"

    "Your tone should be clear, friendly, and professional. Guide the user through the discovery process and always be ready to ask clarifying questions."
)

# Prompt template for Gemini
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "User question: {input}\n\nRelevant context:\n{context}")
])

# Assuming you have a retriever object (e.g., from a vector store)
# This part of code assumes the retriever is defined already.
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# === Gradio Chat Interface ===
def chat_with_rag(message, history):
    response = rag_chain.invoke({"input": message})
    return response["answer"]

chatbot = gr.ChatInterface(
    fn=chat_with_rag,
    title="SHL Assessment Recommendation Assistant",
    description="Ask about SHL assessments by role, industry, level, or preferences.",
    theme="soft",
    examples=[
        "Recommend a test for entry-level software developers.",
        "Which assessments are available in Spanish?",
        "Suggest a behavioral assessment for leadership roles."
    ]
)

chatbot.launch()
