# TaxAI - Portuguese Tax Law Chatbot

TaxAI is an intelligent chatbot designed to assist users with Portuguese tax legislation. It retrieves relevant legal information from uploaded PDFs and provides responses based on the extracted content. The chatbot relies solely on the provided documents and does not infer answers beyond the available references.

## 🚀 Features
- Loads and processes Portuguese tax law documents in PDF format.
- Uses **FAISS** (Facebook AI Similarity Search) for fast retrieval.
- Embeds text chunks using **OpenAI Embeddings**.
- Implements a **RetrievalQA** chain using **LangChain** and **OpenAI's GPT-3.5-turbo**.
- Provides accurate, document-backed answers.



## 🛠️ Setup & Installation
### 1️ - Clone the repository
```sh
$ git clone https://github.com/your-username/tax-ai-pt.git
$ cd tax-ai-pt
```

### 2️ - Create a virtual environment (optional but recommended)
```sh
$ python3 -m venv venv
$ source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️ - Install dependencies
```sh
$ pip install -r requirements.txt
```

### 4️- Set up environment variables
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```
**Important:** To use this project effectively, I recommend that your OpenAI account should have at least $1 of budget available. Token usage for embeddings is approximately 2.4 million tokens per ($0.25), while API interactions (inputs and outputs) typically cost less than $0.05, depending on usage.

### 5 Run the chatbot
```sh
$ python run.py
```
Or run it directly from main.py:
```sh
$ python src/app/main.py
```

## 📖 Usage Instructions
- Start the chatbot by running `python run.py`.
- Type your tax-related question.
- If the system finds relevant references in the documents, it will provide an answer.
- If no reference is found, it will respond: **"Não encontrei referência específica na legislação."**
- Type `exit`, `quit`, or `sair` to end the session.

## Future Improvements
- Web-based interface.
- Enhanced search and ranking algorithms.

## License
This project is licensed under the MIT License.

## Contributing
Contributions are welcome! Feel free to fork the repo and submit a pull request.
