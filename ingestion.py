"""Module for document ingestion into Pinecone vector store."""
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from consts import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    """Load ReadTheDocs documentation, split into chunks, clean metadata, and ingest into Pinecone."""
    # Initialize the ReadTheDocs loader for LangChain API docs
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw documents")

    # Split documents into smaller chunks for efficient embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} document chunks to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("**** Loading to vector store done ***")


def ingest_docs2() -> None:
    """Load documentation via FireCrawlLoader for various LangChain integrations and ingest into Pinecone."""
    from langchain_community.document_loaders.firecrawl import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://python.langchain.com/docs/integrations/chat//",
        "https://python.langchain.com/docs/integrations/llms/",
        "https://python.langchain.com/docs/integrations/text_embedding/",
        "https://python.langchain.com/docs/integrations/document_loaders/",
        "https://python.langchain.com/docs/integrations/document_transformers/",
        "https://python.langchain.com/docs/integrations/vectorstores/",
        "https://python.langchain.com/docs/integrations/retrievers/",
        "https://python.langchain.com/docs/integrations/tools/",
        "https://python.langchain.com/docs/integrations/stores/",
        "https://python.langchain.com/docs/integrations/llm_caching/",
        "https://python.langchain.com/docs/integrations/graphs/",
        "https://python.langchain.com/docs/integrations/memory/",
        "https://python.langchain.com/docs/integrations/callbacks/",
        "https://python.langchain.com/docs/integrations/chat_loaders/",
        "https://python.langchain.com/docs/concepts/",
    ]

    langchain_documents_base_urls2 = [
        "https://python.langchain.com/docs/integrations/chat/"
    ]
    for url in langchain_documents_base_urls2:
        # Log the URL being crawled
        print(f"FireCrawling URL: {url}")
        loader = FireCrawlLoader(
            url=url,
            mode="scrape",
        )
        docs = loader.load()
        # Ingest the scraped documents into the Pinecone index
        print(f"Going to add {len(docs)} documents to Pinecone")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name="firecrawl-index"
        )
        # Confirm loading completion for the crawled URL
        print(f"**** Loading {url} to vector store done ***")


if __name__ == "__main__":
    ingest_docs2()
