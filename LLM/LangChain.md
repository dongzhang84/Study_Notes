
## LangChain Basics — ChatPromptTemplate + LCEL pipe syntax

```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


"""Basic LCEL chain: prompt | model | parser"""
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional sales advisor for B2B software."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"question": "What are the top 3 pain points small businesses face with manual processes?"})
print("=== LCEL Chain Demo ===")
print(response)
print()
```


## Conversation Memory, Uses the modern RunnableWithMessageHistory

```
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

"""Conversation with persistent in-session memory."""
llm = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "demo_session"}}

print("=== Conversation Memory Demo ===")
r1 = conversation.invoke({"input": "My name is John and I run a dental clinic."}, config=config)
print(f"Turn 1: {r1.content}")

r2 = conversation.invoke({"input": "What AI tools might help my business?"}, config=config)
print(f"Turn 2: {r2.content}")

r3 = conversation.invoke({"input": "What is my name and what kind of business do I run?"}, config=config)
print(f"Turn 3 (memory check): {r3.content}")
print()

```



## RAG System — Load, Split, Embed, Store, Retrieve

```
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

  
  

def build_rag_from_pdf(pdf_path: str):

	"""Build a RAG Q&A system from a PDF file with source citations."""
	print(f"=== RAG System — Loading: {pdf_path} ===")
	
	  
	# 1. Load document	
	loader = PyPDFLoader(pdf_path)	
	docs = loader.load()	
	print(f"Loaded {len(docs)} pages.")
	
	  	
	# 2. Split into chunks	
	splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)	
	chunks = splitter.split_documents(docs)	
	print(f"Split into {len(chunks)} chunks.")
		  
	
	# 3. Embed and store in Chroma	
	embeddings = OpenAIEmbeddings()
	vectorstore = Chroma.from_documents(chunks, embeddings)
	  
	
	# 4. Build Q&A chain with source citations	
	qa = RetrievalQA.from_chain_type(	
		llm=ChatOpenAI(model="gpt-4o"),
		retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
		return_source_documents=True,
	)
	
	  
	return qa

  
  

def build_rag_from_text(text: str, source_name: str = "sample_doc"):

	"""Build a RAG Q&A system from a plain text string (for demo without a real PDF)."""
	from langchain_core.documents import Document
		  
	
	print(f"=== RAG System — Building from text ({source_name}) ===")
		  
	
	splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)	
	chunks = splitter.create_documents(
		[text],
		metadatas=[{"source": source_name, "page": i} for i in range(1)]
	)
	
	# Re-split so each chunk gets a page index
	chunks = splitter.split_documents(chunks)	
	for i, chunk in enumerate(chunks):
		chunk.metadata["page"] = i
		
	embeddings = OpenAIEmbeddings()	
	vectorstore = Chroma.from_documents(chunks, embeddings)
		
	qa = RetrievalQA.from_chain_type(	
		llm=ChatOpenAI(model="gpt-4o"),	
		retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),	
		return_source_documents=True,	
	)
	
	return qa

  
  

def ask_with_citations(qa, question: str):

	"""Ask a question and print the answer with source citations."""
	result = qa.invoke(question)
	
	print(f"\nQ: {question}")
	print(f"\nAnswer:\n{result['result']}")
	
	
	print("\nSources:")
	for i, doc in enumerate(result["source_documents"]):
		page = doc.metadata.get("page", "unknown")
		source = doc.metadata.get("source", "unknown")
		snippet = doc.page_content[:120].replace("\n", " ")
		print(f" [{i+1}] {source} — Page {page}: {snippet}...")
	print()

  
  

# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
  

SAMPLE_COMPANY_MANUAL = """

ACME Corp Employee Handbook — Version 3.2


1. COMPANY OVERVIEW

ACME Corp was founded in 2010. We provide B2B software solutions to over 500 clients.

Our headquarters is in San Francisco. We have 120 employees across 3 offices.

  

2. WORKING HOURS

Standard working hours are 9am–6pm, Monday through Friday.

Remote work is permitted up to 3 days per week with manager approval.

Overtime must be pre-approved in writing by your department head.

  

3. VACATION POLICY

Full-time employees receive 15 days of paid vacation per year.

Vacation days accrue at 1.25 days per month.

Unused vacation days can be carried over up to a maximum of 5 days.

Vacation requests must be submitted at least 2 weeks in advance.

  

4. EXPENSE REIMBURSEMENT

Business expenses up to $500 can be self-approved and submitted via the expense portal.

Expenses over $500 require manager pre-approval before purchase.

All receipts must be submitted within 30 days of the expense.

Travel expenses are covered at coach class for flights under 6 hours.

  

5. HEALTH BENEFITS

The company covers 80% of health insurance premiums for employees.

Dependents can be added at the employee's expense.

Dental and vision plans are available as optional add-ons.

Open enrollment is in November each year.

  

6. PERFORMANCE REVIEWS

Performance reviews are conducted twice a year: in June and December.

Salary adjustments are made in January based on the December review.

Employees are rated on a 5-point scale across 4 dimensions.

  

7. REFUND AND RETURN POLICY (for client-facing staff)

Clients may request a full refund within 30 days of purchase.

Refunds for annual subscriptions are prorated after the first month.

Refund requests must include the order number and reason.

Processing time for refunds is 5–10 business days.

  

8. IT AND SECURITY

All employees must complete security training within their first 30 days.

Passwords must be at least 12 characters and changed every 90 days.

Company data must not be stored on personal devices without IT approval.

Report security incidents to security@acmecorp.com within 1 hour of discovery.

"""

  
  

def run_knowledge_base_project():

	"""
	Portfolio Project #1: Company Internal Knowledge Base Q&A System
	- Loads a company manual (text demo; swap for PyPDFLoader on a real PDF)
	- Answers questions with source citations
	"""

	print("=" * 60)
	print("PORTFOLIO PROJECT #1: Company Knowledge Base Q&A System")
	print("=" * 60)

	qa = build_rag_from_text(SAMPLE_COMPANY_MANUAL, source_name="ACME_Handbook_v3.2")
	
	  
	questions = [
		"What is the refund policy for clients?",
		"How many vacation days do employees get per year?",	
		"Can employees work remotely?",	
		"What happens if I need to buy something over $500?",
	]


for question in questions:
	ask_with_citations(qa, question)

   
