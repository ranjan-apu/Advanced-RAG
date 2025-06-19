# Advanced-RAG

Advanced-RAG is a Retrieval-Augmented Generation (RAG) system designed to enhance machine learning models with context-aware responses. This project integrates vector search capabilities with language models to provide accurate and relevant answers based on provided documents and data.

## Overview

This project leverages the following technologies:

- **LangChain**: For integrating vector stores and language models.
- **QdrantVectorStore**: As the vector database for efficient similarity search.
- **MistralAIEmbeddings**: For generating embeddings used in document retrieval.
- **OpenAI API (via OpenRouter)**: For language model capabilities with models like DeepSeek.

The system is designed to prioritize context from retrieved documents while allowing the flexibility to draw on general knowledge when necessary, ensuring comprehensive and accurate responses.

## Installation

### Prerequisites

- Python 3.8 or higher
- Docker (for running Qdrant or other services if needed)
- API keys for MistralAI and OpenRouter

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ranjan-apu/Advanced-RAG.git
   cd Advanced-RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your API keys:
     ```
     MISTRALAI_API_KEY=your_mistralai_api_key
     OPENROUTER_KEY=your_openrouter_api_key
     ```
4. Ensure Qdrant is running locally or update the connection URL in `rag_agent.py` if using a remote instance.

## Usage

1. Load and embed documents into the vector store using `load_embedd_document.py` (adjust as per your document sources):
   ```bash
   python load_embedd_document.py
   ```
2. Run the RAG agent to interact with the system:
   - Modify or extend `rag_agent.py` as needed for your specific use case.
   - The system uses a custom prompt to ensure context-aware responses.

## Project Structure

- `rag_agent.py`: Core script for the RAG system setup with vector store and language model integration.
- `load_embedd_document.py`: Script for loading and embedding documents into the vector store.
- `ai_agent.py`: Additional AI agent functionalities (if applicable).
- `docker-compose.db.yml`: Configuration for database or vector store services.
- `resources/`: Directory containing sample documents or data for embedding.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code adheres to the project's coding standards and includes appropriate documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable, or update as per your licensing choice).

## Contact

For any queries or support, please open an issue on GitHub or contact the repository owner.
