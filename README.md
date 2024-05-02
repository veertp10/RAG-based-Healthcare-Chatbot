# RAG-based-Healthcare-Chatbot
end to end,RAG based healthcare chatbot with multifaceted NLP capabilities

# Project Description
This project is a conversational AI system that can answer questions based on the content of PDF documents. It uses natural language processing techniques like named entity recognition, intent classification, and semantic search to understand user queries and retrieve relevant information from the PDF documents. Additionally, it can also predict potential diseases based on the user's reported symptoms.

## Setup and Run

1. **Create a Conda or Virtual Environment**

   - For Conda:
     ```
     conda create -p venv python==3.10
     ```
   - For Virtualenv:
     ```
     python3 -m venv venv
     ```

2. **Activate the Environment**

   - For Conda:
     ```
     conda activate venv/
     ```
   - For Virtualenv:
     ```
     venv\Scripts\activate (on Windows)
     source venv/bin/activate (on Unix/MacOS)
     ```

3. **Install Dependencies**
    - pip install -r requirement.txt

4. **Set up Google API Key**
- Create a `.env` file in the project root directory.
- Add your Google API key to the `.env` file:
  ```
  GOOGLE_API_KEY=your_google_api_key
  ```

5. **Train the Model**
- Run the `train.py` script from the `src` directory:
  ```
  python src/train.py
  ```

6. **Run the Application**
-   Run the `chatpdf.py` script
    ```
    python chatpdf1.py
    ```
After following these steps, your application should be up and running. You can then access the chatbot interface in your web browser at `http://localhost:8000`.
