import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from datasets import Dataset
from giskard.rag import QATestset

# Load environment variables
load_dotenv()

# Load the Giskard-generated testset
giskard_testset = QATestset.load("my_testset.jsonl")
df_test = giskard_testset.to_pandas()

# Extract questions for evaluation
test_queries = df_test["question"].tolist()

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


# Function to initialize the chatbot
def initialize():
    try:
        llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0.1, presence_penalty=0.5, frequency_penalty=0.1)

        # Define Search
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("Tavily API Key is missing. Please set it in your environment variables.")

        search = TavilySearchResults(api_key=tavily_api_key, max_results=5)

        tools = [
            Tool(
                name="Vector Search",
                func=lambda q: vector_search(q)[0],  # Only return content
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


# Function to evaluate the RAG system using Giskard-generated questions
def evaluate_rag_system(queries):
    responses = []
    retrieved_docs = []

    chatbot_engine = initialize()

    if chatbot_engine is None:
        print("Error: Chatbot engine not initialized.")
        return

    for query in queries:
        try:
            content, sources = vector_search(query)
            prompt = f"You are a consultant providing community healthcare services for seniors. Answer the question based on:\n\n{content}"
            result = chatbot_engine.run(prompt)

            # Store results for evaluation
            responses.append(result)
            retrieved_docs.append(content)

        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    # Convert data to a Hugging Face dataset
    dataset = Dataset.from_dict({
        "question": queries,
        "answer": responses,
        "contexts": [[doc] for doc in retrieved_docs],
        "reference": df_test["reference_answer"].tolist()
    })

    # Define metrics for evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness]
    )

    print("\n=== RAGAS Evaluation Metrics ===")
    print(results)

    return results


# Run evaluation using Giskard-generated test questions
evaluate_rag_system(test_queries)
