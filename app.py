import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain, MapReduceDocumentsChain, StuffDocumentsChain
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import faiss
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI

# Function to check if the text is related to the Constitution
def is_constitution_related(text):
    keywords = ["We the People", "United States", "Constitution", "Amendment", "Senate", "Congress"]
    for keyword in keywords:
        if keyword in text:
            return True
    return False

# Function to split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Function to generate the answer
def ask_question(question):
    result = retriever.run(question)
    return result

# Load the preloaded Constitution PDF
file_path = 'constitution.pdf'  # Replace with the actual path to your preloaded PDF
pdf = PdfReader(file_path)
total_pages = len(pdf.pages)
text = ""
for page_number in range(total_pages):
    page = pdf.pages[page_number]
    text += page.extract_text()

# Check if the text is related to the Constitution
if is_constitution_related(text):
    st.title('American Constitution QA')

    st.markdown("""
    ## Welcome to the American Constitution QA Tool
    This tool is designed to help you ask questions about the American Constitution and receive detailed answers using advanced AI models.

    ### How to Use This Tool
    1. **Review Extracted Text**: The text from the preloaded American Constitution PDF is extracted and displayed below.
    2. **Ask a Question**: Enter your question about the American Constitution in the input box provided.
    3. **Get an Answer**: Click the "Get Answer" button to receive a detailed answer. If you want to regenerate the answer, click the "Regenerate Answer" button.

    ### Purpose of This Tool
    The American Constitution QA Tool leverages the power of AI to provide insightful and accurate answers to questions related to the American Constitution. This tool aims to enhance understanding and accessibility of constitutional knowledge for students, researchers, and anyone interested in learning more about the foundational document of the United States.
    """)

    #st.write("### Extracted Text from Preloaded PDF:")
    #st.write(text)

    # Count the tokens
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(text)
    num_tokens = len(tokens)
    st.write(f"Total number of tokens in the PDF: {num_tokens}")

    chunks = split_text(text, chunk_size=500)

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    documents = {str(i): Document(page_content=chunk) for i, chunk in enumerate(chunks)}
    st.write(f"You now have {len(documents)} docs instead of 1 piece of text")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    docstore = InMemoryDocstore(documents)

    # Map index to docstore IDs
    index_to_docstore_id = {i: str(i) for i in range(len(documents))}

    vector_store = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=model.encode
    )

    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]  # Make sure to add your Google API key in Streamlit secrets
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    qa_template = """
    Question: {question}
    Context: {context}
    Answer:
    """

    prompt = PromptTemplate(template=qa_template, input_variables=["question", "context"])

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    qa_chain = LLMChain(llm=llm, prompt=prompt)

    reduce_documents_chain = StuffDocumentsChain(llm_chain=qa_chain, document_variable_name="context")

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=qa_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="context"
    )

    retriever = RetrievalQA(
        retriever=vector_store.as_retriever(),
        combine_documents_chain=map_reduce_chain
    )

    # User input for question
    question = st.text_input("Enter your question about the American Constitution:")

    if st.button("Get Answer") or st.button("Regenerate Answer"):
        if question:
            with st.spinner('Generating answer...'):
                st.session_state.answer = ask_question(question)
            st.write(f"### Question: {question}")
            st.write(f"### Answer: {st.session_state.answer}")
else:
    st.error("The preloaded document does not appear to be related to the American Constitution.")
