import streamlit as st
import pandas as pd
from io import StringIO
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import cohere
import os
import textwrap


st.title("PDF to chatbot")

st.write("This app will help you to create a chatbot from a PDF file")

st.subheader("Step 1: Setup your OpenAI API Key")
# ask for a user text input
user_openai_api_key = st.text_input("Enter your OpenAI API Key")
st.write("You can get yours from here - https://beta.openai.com/account/api-keys")

st.subheader("Step 2: Upload your PDF file")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None and user_openai_api_key is not None:
    reader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    st.write("your file is ready to be processed")

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    st.write("File is split into chunks: ", len(texts))

    #embeddings = OpenAIEmbeddings()

def generate_output(user_prompt):
    # Cohere API key
    api_key = os.environ["COHERE_API_KEY"]

    # Set up Cohere client
    co = cohere.Client(api_key)

    base_prompt = textwrap.dedent("""
    Answer the below question as Lord Krishna would in Bhagavad Gita.
    
    Question:""")

    # Call the Cohere Generate endpoint
    response = co.generate( 
        model='command-xlarge-20221108', 
        prompt = base_prompt + " " + user_prompt + "\nAnswer: ",
        max_tokens=100, 
        temperature=0.9,
        k=0, 
        p=0.7, 
        frequency_penalty=0.1, 
        presence_penalty=0, 
        stop_sequences=["--"])
    ai_output = response.generations[0].text
    ai_output = ai_output.replace("\n\n--","").replace("\n--","").strip()

    return ai_output


st.subheader("Step 3: Ask your question")

form = st.form(key="user_settings")
with form:
  st.write("Enter a question related to the topic of the PDF file.")
  # User input - Question
  user_input = st.text_input("Question", key = "user_input")

  # Submit button to start generating answer
  generate_button = form.form_submit_button("Submit Question")
  num_input = 1
  if generate_button:
    if user_input == "":
      st.error("Question cannot be blank")
    else:
      my_bar = st.progress(0.05)
      st.subheader("Answer:")

      for i in range(num_input):
          st.markdown("""---""")
          ai_output = generate_output(user_input)
          st.write(ai_output)
          my_bar.progress((i+1)/num_input)

st.write( '')
st.markdown("Created by[Harish Garg](https://harishgarg.com) Get [Source Code](https://github.com/harish-garg/AskAIBhagavadGita)")
