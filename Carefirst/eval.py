import os
import giskard
import pandas as pd
from giskard.rag import QATestset, evaluate, KnowledgeBase
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Load the test set
testset = QATestset.load("my_testset.jsonl")

# Load the knowledge base from `base.csv`
df = pd.read_csv("base.csv")
knowledge_base = KnowledgeBase.from_pandas(df, columns=["question", "answer"])

# Function to perform vector search
def vector_search(query: str) -> tuple:
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    results = retriever.invoke(query)

    # Extract content and sources
    content = "\n".join([doc.page_content for doc in results])
    sources = [doc.metadata.get('source', 'Unknown') for doc in results]

    return content, sources

# Initialize the chatbot
def initialize_chatbot():
    try:
        # Define models
        llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0.1, presence_penalty=0.5, frequency_penalty=0.1)

        # Define Search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("Tavily API Key is missing. Please set it in your environment variables.")

        search = TavilySearchResults(api_key=tavily_api_key, max_results=5)

        tools = [
            Tool(
                name="Vector Search",
                func=lambda q: vector_search(q)[0],  # Only return the content
                description="Used to search for Carefirst Ontario and other Ontario services in the local document database."
            ),
            Tool(
                name="Current Search",
                func=search.run,
                description="Used to search for a product, organization, government policy, eligibility in Ontario."
            )
        ]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chatbot_engine = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            max_tokens=10000,
        )

        return chatbot_engine
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return None

# Function to generate chatbot responses
def get_answer_fn(question: str, history=None) -> str:
    """A function representing the chatbot agent."""
    chatbot_engine = initialize_chatbot()
    
    if chatbot_engine is None:
        return "Error: Chatbot engine is not initialized."

    # Perform the vector search to get potential sources
    content, sources = vector_search(question.strip())

    # Format the query with relevant content
    prompt = f"""You are a Carefirst Ontario services consultant. You are careful and attentive to details. Most clients are Chinese-speaking seniors, primarily in Scarborough, Markham, Richmond Hill, Newmarket, North York, and Downtown Toronto. ***Note: Scarborough is NOT part of Durham Region!***

    ### User Query:
    {question}

    ### Relevant Content:
    {content}

    ### Sources:
    {"\n".join(f"- {source}" for source in sources)}
    """

    # Get chatbot response
    response = chatbot_engine.run(prompt)
    return response

# Run the evaluation and generate a report
report = evaluate(get_answer_fn, testset=testset, knowledge_base=knowledge_base)

# Save the report as an HTML file
report.to_html("rag_eval_report.html")
print("✅ RAG evaluation report saved as 'rag_eval_report.html'.")

