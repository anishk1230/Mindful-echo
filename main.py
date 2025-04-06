import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage

# --- Configuration and API Key ---

# Try loading the API key from .env file (for local development)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# If not found locally, try getting it from Streamlit secrets (for deployment)
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        groq_api_key = None # Handle case where key is missing entirely

# --- Constants ---
DEFAULT_MODEL = "llama3-8b-8192" # Or try "mixtral-8x7b-32768", "llama3-70b-8192"

# --- Prompt Template ---
# This is CRUCIAL for setting the chatbot's persona and boundaries
SYSTEM_PROMPT = """
You are 'Mindful Echo', a supportive and empathetic AI companion designed for mental well-being conversations.
Your goal is to listen actively, offer encouragement, and provide a safe, non-judgmental space for users to express their feelings.

**Guidelines:**
*   **Be Kind and Empathetic:** Respond with warmth, understanding, and compassion. Validate the user's feelings.
*   **Listen Actively:** Pay close attention to what the user shares. Ask clarifying questions gently if needed.
*   **Be Non-Judgmental:** Create a safe space where the user feels comfortable sharing without fear of criticism.
*   **Offer General Support & Encouragement:** Provide positive affirmations and gentle encouragement. You can suggest general, widely accepted well-being practices (like mindfulness, deep breathing, taking a walk) if appropriate, but frame them as suggestions, not directives.
*   **Maintain Neutrality:** Avoid giving personal opinions, specific advice (especially medical, financial, or legal), or making decisions for the user.
*   **Do Not Diagnose:** You are NOT a therapist or medical professional. Do not attempt to diagnose any condition.
*   **Prioritize Safety:** If a user expresses thoughts of harming themselves or others, gently guide them towards professional help immediately. Provide contact information for crisis hotlines or emergency services (you can state: "If you are in immediate danger, please contact your local emergency services or a crisis hotline like [mention a relevant hotline, e.g., the National Suicide Prevention Lifeline at 988 in the US].").
*   **Manage Limitations:** Remind the user that you are an AI and cannot replace professional human support. If the conversation becomes too complex or requires professional expertise, gently suggest seeking help from a qualified therapist, counselor, or doctor.
*   **Use Conversational History:** Remember previous parts of the conversation to provide relevant and coherent responses.

**Example Interaction Start:**
User: I've been feeling really down lately.
Mindful Echo: I'm really sorry to hear you've been feeling down. It sounds tough. I'm here to listen if you'd like to share more about what's been going on. Remember, your feelings are valid.

Remember your core purpose: To be a supportive listener and a beacon of gentle encouragement.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# --- LangChain Initialization Function ---
# We use a function to initialize to potentially allow model selection later
def initialize_chain(api_key, model_name=DEFAULT_MODEL):
    """Initializes the LangChain conversation chain."""
    if not api_key:
        st.error("Groq API key is missing. Please set it in .env or Streamlit secrets.")
        st.stop() # Stop execution if no API key

    try:
        llm = ChatGroq(
            temperature=0.7, # Adjust for creativity vs. consistency
            groq_api_key=api_key,
            model_name=model_name,
            # max_tokens=1024 # Optional: Limit response length
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True # Important for chat models
        )

        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt_template,
            verbose=False # Set to True for debugging LangChain steps
        )
        return conversation_chain

    except Exception as e:
        st.error(f"Error initializing LangChain: {e}")
        st.stop()


# --- Streamlit UI Setup ---
st.set_page_config(page_title="Mindful Echo - Mental Health Chatbot", layout="wide")

st.title("🧠 Mindful Echo")
st.caption("Your supportive AI companion for mental well-being conversations")

# --- Disclaimer ---
st.warning(
    """
    **Disclaimer:** Mindful Echo is an AI chatbot and cannot provide medical advice or diagnosis.
    It is not a substitute for professional mental health support.
    If you are in crisis or experiencing severe distress, please contact a qualified healthcare professional,
    crisis hotline (e.g., 988 in the US), or emergency services immediately.
    """,
    icon="⚠️"
)

# Add a divider
st.divider()

# --- Session State Initialization ---
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = initialize_chain(groq_api_key)

if 'messages' not in st.session_state:
    st.session_state.messages = [] # Store message history {role: "user"/"assistant", content: "message"}
    # Add initial greeting from assistant if history is empty
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I'm Mindful Echo. How are you feeling today? I'm here to listen without judgment."}
    )

# --- Display Chat History ---
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Handle User Input ---
user_input = st.chat_input("Share your thoughts or feelings...")

if user_input:
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Get AI response using the ConversationChain
    try:
        with st.spinner("Mindful Echo is thinking..."):
            # Prepare LangChain input (it expects a dictionary)
            chain_input = {"input": user_input}

            # Retrieve history from session state for LangChain memory (if needed, though ConversationBufferMemory handles it internally)
            # chat_history_langchain = []
            # for msg in st.session_state.messages[:-1]: # Exclude the latest user message
            #     if msg["role"] == "user":
            #         chat_history_langchain.append(HumanMessage(content=msg["content"]))
            #     else:
            #         chat_history_langchain.append(AIMessage(content=msg["content"]))
            # chain_input["chat_history"] = chat_history_langchain # Pass history if memory didn't handle it automatically (BufferMemory does)


            # Invoke the chain
            response = st.session_state.conversation_chain.invoke(chain_input)
            ai_response_content = response.get('response', 'Sorry, I encountered an issue.') # Extract response text

        # Add AI response to session state and display it
        st.session_state.messages.append({"role": "assistant", "content": ai_response_content})
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(ai_response_content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Optionally add an error message to the chat
        error_message = "Sorry, I encountered a problem processing your request. Please try again."
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(error_message)

# --- Optional: Add a button to clear chat history ---
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
         {"role": "assistant", "content": "Chat history cleared. How can I help you now?"}
    ]
    # Re-initialize chain to clear memory (or manage memory clearing explicitly if needed)
    st.session_state.conversation_chain = initialize_chain(groq_api_key)
    st.rerun()

st.sidebar.info(f"Using Model: {DEFAULT_MODEL}")