from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from decouple import config
import requests # Replace with the actual Gemini API client library

# Set up Gemini API client
gemini_client = genai.GenerativeModel('gemini-pro')

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="../vector_db",
    collection_name="rich_dad_poor_dad",
    embedding_function=embedding_function,
)

# create prompt
QA_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.

chat_history: {chat_history}

Context: {text}

Question: {question}

Answer:""",
    input_variables=["text", "question", "chat_history"]
)

# create memory
memory = ConversationBufferMemory(
    return_messages=True, memory_key="chat_history")

# question
question = "What is the book about?"

def rag(question: str) -> str:
    # Call Gemini API for question answering
    response = gemini_client.question_answering(question, context=vector_db.texts)
    return response

# Example usage
print(rag(question))