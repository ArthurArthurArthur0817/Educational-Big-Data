import os
import sys
import textwrap
import time
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import torch
from langchain_community.vectorstores import FAISS
import pandas as pd

# 读取命令行参数
question_path = sys.argv[1]
pdf_paths = sys.argv[2:]

# 确认文件路径
print("Received Question Path:", question_path)
print("Received PDF Paths:", pdf_paths)

class CFG:
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    temperature = 0.5
    top_p = 0.95
    repetition_penalty = 1.15
    do_sample = True
    max_new_tokens = 400

    split_chunk_size = 800
    split_overlap = 0
    
    embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'
    k = 3
    PDFs_paths = pdf_paths
    Embeddings_path = './faiss_index_py'

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# 明确地传递模型参数
llm = HuggingFaceEndpoint(
    repo_id=CFG.model_name,
    max_new_tokens=CFG.max_new_tokens,
    temperature=CFG.temperature,
    top_p=CFG.top_p,
    repetition_penalty=CFG.repetition_penalty,
    do_sample=CFG.do_sample
) 

documents = []
for pdf_path in CFG.PDFs_paths:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    else:
        print(f"File not found: {pdf_path}")
        sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CFG.split_chunk_size,
    chunk_overlap=CFG.split_overlap
)

texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceInstructEmbeddings(
    model_name=CFG.embeddings_model_repo
)

vectordb = FAISS.from_documents(
    documents=texts, 
    embedding=embeddings
)

vectordb.save_local("faiss_index_py")

embeddings = HuggingFaceInstructEmbeddings(
    model_name=CFG.embeddings_model_repo
)

vectordb = FAISS.load_local(
    CFG.Embeddings_path,
    embeddings
)

prompt_template = """
<s>[INST] 
Do not attempt to fabricate an answer. If the information is not available in the context, simply state that you don't know.
Answer in the same language the question was asked.
Provide a concise and accurate answer based strictly on the provided context.
Reference specific chemical theories, formulas, or data directly from the context.
Use technical and professional language suitable for a chemistry research paper.
Ensure the answer is precise and clearly references the provided context.
Quote specific sentences or data points from the context where applicable.
If the question cannot be answered based on the provided context, state explicitly that the information is not available in the provided context.

{context}

Question: {question}
Answer:[/INST]"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["question", "context"]
)
llm_chain = LLMChain(prompt=PROMPT, llm=llm)
retriever = vectordb.as_retriever(search_kwargs={"k": CFG.k, "search_type": "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever, 
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True,
    verbose=False
)

def wrap_text_preserve_newlines(text, width=700):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4] + ' - page: ' + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    ans = ans + '\n\nSources: \n' + sources_used
    print("=================================")
    print(sources_used)
    return ans

def llm_ans(query):
    start = time.time()
    llm_response = qa_chain(query)
    ans = process_llm_response(llm_response)
    end = time.time()
    time_elapsed = int(round(end - start, 0))
    time_elapsed_str = f'\n\nTime elapsed: {time_elapsed} s'
    return ans.strip() + time_elapsed_str

def extract_text_after_inst(input_string):
    marker_index = input_string.find("[/INST]")
    if (marker_index != -1):
        return input_string[marker_index + len("[/INST]"):].strip()
    else:
        return ""

data = pd.read_excel(question_path)

def predict(message, data_content=None):
    if data_content:
        message += f"\nData: {data_content}"
    output = str(llm_ans(message))
    output = extract_text_after_inst(output)
    return output

for index, row in data.iterrows():
    question = row['question']
    data_content = row.get('data', None)
    answer = predict(question, data_content)
    data.at[index, 'answer'] = answer

data.to_excel("answers.xlsx", index=False)
