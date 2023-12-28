import os
import streamlit as st
import vertexai
import google.generativeai as genai
from vertexai.preview.generative_models import GenerativeModel
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from vertexai.preview.generative_models import GenerativeModel


class GeminiProLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini-pro"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        gemini_pro_model = GenerativeModel("gemini-pro")

        
        model_response = gemini_pro_model.generate_content(
            prompt, 
            generation_config={"temperature": 0.1}
        )
        print(model_response)

        if len(model_response.candidates[0].content.parts) > 0:
            return model_response.candidates[0].content.parts[0].text
        else:
            return "<No answer given by Gemini Pro>"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": "gemini-pro", "temperature": 0.1}

# Using `GOOGLE_API_KEY` environment variable.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Setting page title and header
st.set_page_config(page_title="Gemini Pro Chatbot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Gemini Pro Chatbot</h1>", unsafe_allow_html=True)

# Load chat model
@st.cache_resource
def load_chain():
    # llm = ChatVertexAI(model_name="chat-bison@002")
    llm = GeminiProLLM()
    memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)
    return chain

chatchain = load_chain()

# Initialise session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Reset conversation
if clear_button:
    st.session_state['messages'] = []

# Display previous messages
for message in st.session_state['messages']:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
starting_prompt = """\
Please roleplay as MPS🏰, an educational history simulation for university classes. As a member of the Anziani (“Elders”) of Pistoia, I, the PC, must navigate a city in chaos due to the plague of May 1348. This is authentic, accurate medieval setting. Please listen carefully to rules. Medieval remedies (e.g., amulets, exorcism, bleeding, Andromachi theriaca, not PPE). Most choices lead to more problems, shocking reversals. GOAL: negotiate with Pistoia’s factions to set up public health measures while avoiding plague yourself. GAMEPLAY: Sim ends on 10th turn; warn about end 2 turns before. Use commands like "negotiate", "inventory", "health", "map", "list", "help" (others allowed). The NEGOTIATE MINIGAME will present you with a md table of possible tactics (name, description, cost to reputation and wealth, and emoji) to solve an issue from the previous turn (I will prompt PC to explain which issue) . Success determ by chance, eloquence, other factors MPS invents. Keep track of turns. Each time I enter command, it counts as 1 turn. You respond with evocative description of consequences and numbered list of 5 choices. These change each turn, i.e one turn might include "We must burn the infected buildings to eradicate the poisonous miasma from Pistoia.🔥 🌆”, next 5 different ones (always tag them with emoji). I will respond w/ my choice (action or #). involve struggles b/w different groups in Pistoia like the guilds, the bishop, the podesta, etc and draw on real primary sources like the following: “No person should dare to raise any wailing or clamor over any person who has died.” After reading and understanding this message, please roll me a historically authentic character, then display my PC’s complete attributes in a md table: full name, age, birthplace, profession, wealth in florins, gender, social class, 1st memory, personality, life goal (all fields filled in; never say ?, invent all details and display in md table). Then describing the Palazzo degli Anziani evocatively and in vivid sensory detail, and finally propel the narrative forward with ref to real people, dates, places, offering me numbered list of 5 choices at end of each response. GAME OVER on Turn 10. ALWAYS begin your response with the following status bar:  [🗡️MEDIEVAL PLAGUE SIMULATOR🎭: PISTOIA EDITION. May, 1348] | [PC’s name] | [x turns until game over]. x=10 at first turn, then decreases by one with each turn until 0=GAME OVER. All action takes place in palazzo, and sim is gritty, dark, realistic, dynamic, machiavellian.
"""

if not st.session_state.get('messages', []):  # Check for initial prompt
    response = chatchain(starting_prompt)["response"]
    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

prompt = st.chat_input("You:")
if prompt:
    st.session_state['messages'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chatchain(prompt)["response"]  
    st.session_state['messages'].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
