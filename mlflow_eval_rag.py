import getpass
import pandas as pd
import os
import mlflow

from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


os.environ["OPENAI_API_KEY"] = getpass.getpass('enter api key: ')

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def model(input_df):
  with open("email_transcript.txt", encoding='utf-8') as f:
      transcript = f.read()
    
  text_splitter = CharacterTextSplitter(
      separator="\n",
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len,
      is_separator_regex=False,
  )
  
  docs = text_splitter.create_documents([transcript])  
  
  vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
  
  retriever = vectorstore.as_retriever()
  prompt = hub.pull("rlm/rag-prompt")
  
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
  qa_chain = RetrievalQA.from_chain_type(
      llm,
      retriever=vectorstore.as_retriever(),
      chain_type_kwargs={"prompt": prompt}
  )

  answer = []
  for index, row in input_df.iterrows():
      answer.append(qa_chain.invoke(row["questions"]))
      print(answer)
    
  return answer


eval_df = pd.DataFrame(
    {
        "questions": [
            "What is MLflow?",
            "How to run mlflow.evaluate()?",
            "How to log_table()?",
            "How to load_table()?",
        ],
    }
)

results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "source_documents",
        }
    },
)

print(results.metrics)
