import os
import chainlit as cl
from google.genai import Client
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot"

client = Client(
  http_options= {'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"

config = {
  "generation_config": {"response_modalities": ["TEXT"]}
  }

memory = MemorySaver()

# State

class State(MessagesState):
    pass

# Invoke Messages

async def greet_user(state: State):
    # Ensure state contains 'messages'
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")
    
    # Extract the last user message from the state
    messages = state["messages"]
    last_message = messages[-1].content.lower()


    # Define a prompt to identify the greeting intent and respond dynamically
    intent_prompt = (
        f"The user said: '{last_message}'. "
        "If this is a greeting, respond with an appropriate greeting. "
        "If it's not a greeting, respond with: 'I only handle greetings right now.'"
    )

    # Call the LLM dynamically to get the response
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            # Send the prompt to the LLM
            await session.send(input=intent_prompt, end_of_turn=True)
    
            # Stream the model's response
            response_text = ""
            turn = session.receive()  # session.receive() is an async generator, don't await here
            async for chunk in turn:  # Use async for to iterate over the generator
                if chunk.text is not None:
                    response_text += chunk.text

    except Exception as e:
        raise RuntimeError(f"Error invoking the model: {e}")

    # Append the LLM's response to the state messages
    response_message = AIMessage(content=response_text)
    messages.append(response_message)

    # Update and return the state
    return {"messages": messages[-1]}

# Build Graph

builder = StateGraph(State)

builder.add_node("greet", greet_user)

builder.add_edge(START, "greet")
builder.add_edge("greet", END)

graph = builder.compile(checkpointer=memory)


#### Chainlit

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("Rida Naz", "user"):
        return cl.User(
            identifier="Rida Naz", metadata={"role": "user", "provider": "credentials"}
        )
    else:
        return None

@cl.on_message
async def on_message(msg: cl.Message):
    config = {
        "configurable": {"thread_id": cl.context.session.id},
        "generation_config": {"response_modalities": ["TEXT"]}
        }
    
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    # Stream the conversation using the graph
    async for response_msg, metadata in graph.astream(
        {"messages": [HumanMessage(content=msg.content)]},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        # Check if the response is from the agent and not a ToolMessage
        if response_msg.content and isinstance(response_msg, AIMessage):
            # Stream only the agent's response
            await final_answer.stream_token(response_msg.content)

    # Send the final answer once streaming is complete
    await final_answer.send()