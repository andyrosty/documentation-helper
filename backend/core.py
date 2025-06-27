"""Core module for executing retrieval-augmented generation (RAG) chains using LangChain and Pinecone."""
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from typing import Any, Dict, List

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

from consts import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """Execute a history-aware retrieval-augmented generation chain.
    Args:
        query: The user's input question.
        chat_history: List of role-content tuples for conversation history.
    Returns:
        A dict with 'answer' (model response) and 'context' (retrieved documents).
    """
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


def format_docs(docs):
    """Concatenate document contents separated by double newlines.
    Args:
        docs: Iterable of document objects with page_content attribute.
    Returns:
        A single string combining each document's page_content.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    """Run an alternative RAG chain pipeline using RunnablePassthrough and output parser.
    Args:
        query: The user's input question.
        chat_history: Conversation history as a list of role-content pairs.
    Returns:
        Parsed model output after retrieval and generation.
    """
    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    chat = ChatOpenAI(model_name="gpt-4o", verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    rag_chain = (
        {
            "context": docsearch.as_retriever() | format_docs,
            "input": RunnablePassthrough(),
        }
        | retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()

    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(
        answer=rag_chain
    )

    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result
