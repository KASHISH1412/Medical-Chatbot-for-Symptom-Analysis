import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Optional: Uncomment if using .env files
from dotenv import load_dotenv
load_dotenv()

# Hugging Face Token & Model
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Vector DB
DB_FAISS_PATH = r"C:\Users\HP\Desktop\Project\medical-chatbot-main\vectorstore\db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS database loaded successfully!")
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    exit(1)

# Create QA chain
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    print("QA chain created successfully!")
except Exception as e:
    print(f"Error creating QA chain: {e}")
    exit(1)

# Query Loop
def main():
    print("RAG System with Llama 3.1 8B Instruct")
    print("Type 'quit' or 'exit' to stop")
    print("-" * 50)
    
    while True:
        user_query = input("\nWrite Query Here: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not user_query:
            print("Please enter a valid query.")
            continue
            
        try:
            print("Processing your query...")
            response = qa_chain.invoke({'query': user_query})
            
            print("\n" + "="*50)
            print("RESULT:")
            print("="*50)
            print(response["result"])
            
            print("\n" + "="*50)
            print("SOURCE DOCUMENTS:")
            print("="*50)
            for i, doc in enumerate(response["source_documents"], 1):
                print(f"\nSource {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"Metadata: {doc.metadata}")
            print("="*50)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            print("Please try again with a different query.")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not found!")
        print("Please set your Hugging Face token as an environment variable.")
        exit(1)
    
    main()
