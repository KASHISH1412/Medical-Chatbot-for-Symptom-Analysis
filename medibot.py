import streamlit as st
import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


@st.cache_resource
def load_llm():
    """Load LLM using FREE Google Gemini API"""
    try:
        # Configure Gemini API
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Free tier model
            temperature=0.5,
            max_tokens=512,
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
        st.success("✅ Google Gemini API connected successfully!")
        return llm
    except Exception as e:
        st.error(f"❌ Error connecting to Gemini API: {str(e)}")
        return None


def main():
    st.title("Medical Chatbot - Ask Me Anything!")
    st.markdown("*Powered by Google Gemini (FREE)*")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    if prompt := st.chat_input("Ask your medical question here..."):
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Configuration
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's medical question.
        If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
        Don't provide anything outside of the given context.
        Provide accurate medical information based on the context provided.

        Context: {context}
        Question: {question}

        Answer directly and professionally:
        """

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Initialize QA chain if not already done
                    if st.session_state.qa_chain is None:
                        vectorstore = get_vectorstore()
                        if vectorstore is None:
                            st.error("Failed to load the vector store")
                            return

                        llm = load_llm()
                        if llm is None:
                            st.error("Failed to load the language model")
                            return

                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(
                                search_kwargs={'k': 3}),
                            return_source_documents=True,
                            chain_type_kwargs={
                                'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                        )

                    # Get response
                    response = st.session_state.qa_chain.invoke(
                        {'query': prompt})

                    result = response["result"]
                    source_documents = response["source_documents"]

                    # Display the main answer
                    st.markdown(result)

                    # Display source documents in an expander
                    if source_documents:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(source_documents, 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(doc.page_content[:300] + "...")
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.markdown(f"*Metadata: {doc.metadata}*")
                                st.markdown("---")

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': result})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {'role': 'assistant', 'content': error_msg})

    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            "This medical chatbot uses Google Gemini (FREE) to answer questions based on medical documents.")

        st.markdown("### API Status")
        if os.environ.get("GOOGLE_API_KEY"):
            st.success("✅ Google API Key configured")
        else:
            st.error("❌ Google API Key missing")

        st.markdown("### Free Tier Limits")
        st.info("📊 Gemini 1.5 Flash: 15 requests/minute, 1M requests/day")

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.qa_chain = None
            st.rerun()


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY environment variable not found!")
        st.markdown("**Get your FREE Google API key:**")
        st.markdown(
            "1. Go to [Google AI Studio](https://aistudio.google.com/)")
        st.markdown("2. Create a free account")
        st.markdown("3. Generate an API key")
        st.markdown("4. Add to your `.env` file:")
        st.code("GOOGLE_API_KEY=your_api_key_here")
        st.stop()

    main()