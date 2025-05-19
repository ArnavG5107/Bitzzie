import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration for a wider layout and custom theme
st.set_page_config(
    page_title="BITZZIE AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Custom CSS for appearance with black background and colored accents
st.markdown("""
    <style>
    .main {
        background-color: #FFE6E6; /* Blush color background */
        padding: 20px;
        border-radius: 10px;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #E0E0E0;
    }
    .user-message {
        background-color: Black;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 8px solid red;
    }
    .bot-message {
        background-color: Black;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid #7E57C2;
    }
    .title-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    /* Simple fix for model dropdown color */
    .element-container:has(#model_selection) {
        color: white !important;
    }
    .user-message b, .bot-message b {
        color: white;
    }
    /* Hide hamburger menu and toolbar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    /* Hide deploy and reset buttons */
    .stDeployButton {display: none;}
    button[kind="primaryFormSubmit"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please set the GROQ_API_KEY environment variable")
        return

    # Create a header with logo and title - swap positions (logo on left)
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image('BITZZIE AI.png', width=120)
    with col2:
        st.markdown("<h1 style='color: #7E57C2;'>BITZZIE AI</h1>", unsafe_allow_html=True)
    
    # Introduction message - fixed formatting
    st.markdown("""
    <div class="bot-message">
    Hello! I'm your friendly AI assistant, BITZZIE. I can help answer your questions, provide information, or just chat. 
    Let's start our conversation!
    </div>
    """, unsafe_allow_html=True)

    # Add only model selection to the sidebar - hide other customization options
    with st.sidebar:
        st.markdown("<h2 style='color: #7E57C2;'>SELECT MODEL</h2>", unsafe_allow_html=True)
        
        # Hidden system prompt (not shown to users but used in the background)
        system_prompt = "You are BITZZIE, a helpful, friendly AI assistant. You provide clear, concise, and accurate information with a touch of personality. You're powered by Groq's fast language models."
        
        # Only show model selection to users with simple styling
        model = st.selectbox(
            'Choose your model:',
            ['llama3-8b-8192', 'gemma2-9b-it'],
            key="model_selection"
        )
        
        # Hidden memory setting (not shown to users but used in the background)
        conversational_memory_length = 50

    # Initialize conversation memory if not present in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create a container for the chat history display
    chat_container = st.container()
    
    # Input for user's question
    user_question = st.text_input("Ask BITZZIE anything:", key="user_input")

    # Import necessary LangChain components here to avoid potential circular imports
    from langchain.chains import ConversationChain, LLMChain
    from langchain_core.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_core.messages import SystemMessage
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    from langchain_groq import ChatGroq

    # Initialize conversation memory
    memory = ConversationBufferWindowMemory(
        k=conversational_memory_length, 
        memory_key="chat_history", 
        return_messages=True
    )
    
    # Load conversation history into memory
    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
        )

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # Process user question if provided
    if user_question:
        # Construct chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )
        
        # Get response from the model
        with st.spinner("BITZZIE is thinking..."):
            response = conversation.predict(human_input=user_question)
            
            # Clean the response to remove any stray HTML tags
            import re
            response = re.sub(r'<\/?p>', '', response)  # Remove any <p> or </p> tags
            response = response.strip()  # Remove extra whitespace
        
        # Save the interaction to chat history
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)

    # Display chat history - Using div instead of p tags for better control
    with chat_container:
        for message in st.session_state.chat_history:
            st.markdown(f"""
            <div class="user-message">
                <b>You:</b> {message['human']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="bot-message">
                <b>BITZZIE:</b> {message['AI']}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
