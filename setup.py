# setup.py
from setuptools import setup, find_packages

setup(
    name="financial_qa_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.10",
        "chromadb>=0.4.15",
        "langchain-chroma>=0.0.5",
        "openai>=1.3.0",
        "python-dotenv>=0.19.0",
        "nest-asyncio>=1.5.6",
        "pandas>=1.3.0",
        "typing>=3.7.4",
        "ipython>=8.0.0"
    ],
)