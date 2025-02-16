# Financial QA RAG System

A question-answering system built with LangChain and Streamlit that combines text and tabular data analysis for financial documents. The system uses RAG (Retrieval Augmented Generation) to provide accurate answers to questions about financial data.

## Features

- Natural language querying of financial documents
- Combined analysis of text and tabular data
- Source document tracking
- Interactive web interface
- Multi-vector retrieval system

## Prerequisites

- Python 3.9 or higher
- OpenAI API key

## Installation

### Clone the Repository

```bash
git clone https://github.com/factorwonk/financial-qa-rag.git
cd financial-qa-rag
```

### Set Up Virtual Environment

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies for Sample Notebooks will not be installed!

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-key-here
```

## Running the Application

1. Make sure your virtual environment is activated

2. Start the Streamlit app:
```bash
streamlit run src/app/main.py
```

3. Open your web browser and navigate to:
```
http://localhost:8501
```

## Project Structure

- `src/app/`: Contains the Streamlit web application
- `src/rag/`: Contains the RAG system implementation
- `data/`: Directory for storing financial documents
- `tests/`: Test files (to be implemented)
- `sample_notebooks/`: end-to-end pipeline; ragas evaluation; streamlit app and helper .py files

## Usage

1. Uses the convfinqa json document in the data folder or get the training data from [here] (https://github.com/czyssrs/ConvFinQA)
2. Start the application using the instructions above
3. Enter your questions in the text input field
4. View answers and source documents in the interface

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Powered by OpenAI's GPT-4
- Frontend created with Streamlit