# Import necessary libraries for text loading, vectorization, embeddings,
# LLM model, and question answering chain.
from langchain.document_loaders import WhatsAppChatLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os

# Set OpenAI API key as an environment variable.
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

# Specify the path to the WhatsApp chat file.
chat_path = "path/to/your/whatsapp_chat.txt"

# Create a WhatsAppChatLoader object with the chat path.
loader = WhatsAppChatLoader(chat_path)

# Load the chat messages.
messages = loader.load()

# Create an OpenAIEmbeddings object for embeddings.
embeddings = OpenAIEmbeddings()

# Create a Chroma object from documents for vectorization, and transform it into a document retriever.
chatsearch = Chroma.from_documents(messages, embeddings).as_retriever()

# Define the query for the model.
query = "What does the chat say about cats"

# Retrieve the documents relevant to the query.
docs = chatsearch.get_relevant_documents(query)

# Load a question answering chain with OpenAI's model (temperature of 0 means deterministic responses).
# Chain_type is set to "stuff".
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# Run the question answering chain with the relevant documents and the query.
output = chain.run(input_documents=docs, question=query)

# Print the output from the question answering chain.
print(output)
