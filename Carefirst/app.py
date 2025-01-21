import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


# Function to set API Key and initialize chatbot
def set_key(api_key):
    if not api_key:
        return "Key can't be empty!", None

    os.environ["OPENAI_API_KEY"] = api_key

    chatbot_engine = initialize()
    if chatbot_engine is None:
        return "Error initializing chatbot. Check logs.", None

    return "Key received! Please start chatting.", chatbot_engine


# Function to create a structured prompt
def create_prompt_template(content: str, sources: list, user_query: str) -> str:
    sources_str = '\n'.join(f"- {source}" for source in sources)
    prompt = f"""You are a Carefirst Ontario services consultant. You are careful and attentive to details. Most clients are Chinese-speaking seniors, primarily in Scarborough, Markham, Richmond Hill, Newmarket, North York, and Downtown Toronto. ***Note: Scarborough is NOT part of Durham Region!***
    
### User Query:
{user_query}

### Relevant Content:
{content}

### Sources:
{sources_str}
"""
    return prompt


# Chat function
def chat(chat_history, chatbot_engine, message=""):
    if chatbot_engine is None:
        return chat_history, chat_history, "Error: Chatbot engine is not initialized. Please set the API key first."

    if not message.strip():
        return chat_history, chat_history, ""

    try:
        # Perform the vector search to get potential sources
        content, sources = vector_search(message.strip())

        # Create the prompt using the template
        prompt = create_prompt_template(content, sources, message.strip())

        # Run the LLM with this new structured prompt
        result = chatbot_engine.run(prompt)

        # Append sources to the result
        result += f"\n\nSources: {', '.join(sources)}"

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    chat_history.append((message, result))
    return chat_history, chat_history, ""


# Clear chat function
def clear_chat(chatbot_engine):
    new_engine = initialize()  # Reinitialize the chatbot engine
    print("Memory after clear:", new_engine.memory.chat_memory if new_engine else "Error initializing")
    return [], [], "", new_engine


# Gradio App
with gr.Blocks() as demo:
    chat_history = gr.State([])  # Chat history state
    chatbot_engine = gr.State(None)  # Initialize chatbot engine as None

    gr.Image("./carefirst_logo.png", show_label=False, width=200)

    with gr.Row():
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key",
            show_label=False,
            lines=1,
            type="password",
        )
        api_key_set = gr.Button("Set key")

    api_key_set.click(
        fn=set_key,
        inputs=[openai_api_key_textbox],
        outputs=[api_key_set, chatbot_engine],
        show_progress=True  # Ensure the UI waits for completion
    )

    gr.Markdown("<h1><center>Chat with Carefirst bot!</center></h1>")
    chatbot = gr.Chatbot(show_copy_button=True)
    message = gr.Textbox(placeholder="How can I help you?")

    with gr.Row():
        submit = gr.Button("SEND")
        clear = gr.Button("Clear Chat")

    submit.click(chat, inputs=[chat_history, chatbot_engine, message], outputs=[chatbot, chat_history, message])
    clear.click(fn=clear_chat, inputs=[chatbot_engine], outputs=[chatbot, chat_history, message, chatbot_engine])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)  # Ensure public access with share=True
