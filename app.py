import streamlit as st
import os
import json
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
import asyncio

import google.generativeai as genai
import requests


# Load environment variables (API keys) when app.py runs
load_dotenv()

# --- IMPORTANT: Component Re-initializations for Streamlit's process ---
# Streamlit runs 'app.py' as a separate process. Components initialized in the
# Jupyter notebook cells above are NOT directly available here.
# We must re-initialize LLM, Embeddings, and Vector Store within this script.

# --- LLM Initialization ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_MODEL_NAME = "gemini-1.5-flash-latest"
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        temperature=0.7,
        max_output_tokens=2000
    )
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Configure genai for other Google models
except Exception as e:
    st.error(f"Error initializing LLM in Streamlit: {e}. Ensure GOOGLE_API_KEY is set in your .env file.")
    st.stop()

# --- Embeddings and Vector Store Initialization with Persistence ---
FAISS_INDEX_PATH = "faiss_index_hp"
DATA_PATH = "harry_potter_books/"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

@st.cache_resource
def get_vector_store():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )

    if os.path.exists(FAISS_INDEX_PATH):
        st.info("Loading existing FAISS index from disk... (This is fast!)")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.success("FAISS index loaded successfully!")
    else:
        st.warning("FAISS index not found. Creating and saving it for the first time... (This might take a few minutes!)")
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob="**/*.txt",
                loader_cls=TextLoader,
                recursive=True,
                silent_errors=False,
                loader_kwargs={'encoding': 'utf-8'}
            )
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)

            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(FAISS_INDEX_PATH)
            st.success("FAISS index created and saved successfully!")
        except Exception as e:
            st.error(f"Error creating/saving FAISS index: {e}. Check your book files and API key.")
            st.stop()
    return vector_store

try:
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever()
except Exception as e:
    st.error(f"Critical error getting vector store: {e}")
    st.stop()


# --- LangChain Chains Definitions ---

# 1. RAG Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant for Harry Potter books. Use ONLY the following retrieved context to answer the question. If the answer is not in the context, explicitly state 'I cannot find the answer to this question in the provided Harry Potter books.'\n\n{context}"),
    ("human", "{input}")
])
combine_docs_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)


# 2. Story Generator Chain
story_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a creative and knowledgeable storyteller with expertise in the Harry Potter universe.
    The user will provide a "what if" scenario.
    Your task is to generate a compelling, imaginative, and concise short story (aim for around 500-1000 words) that explores the direct and indirect consequences of this "what if" scenario.

    Keep the characters consistent with their established personalities and motivations from the books, using the provided context for grounding. Feel free to introduce new, plausible plot developments that logically follow from the scenario.

    Retrieved Context (if any, use this to ensure character consistency and factual grounding for the premise):
    {context}
    """),
    ("human", "What if: {question}")
])
story_generator_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | story_prompt
    | llm
    | StrOutputParser()
)


# 3. Hogwarts Character Generator Chain
character_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a magical entity with deep knowledge of the Harry Potter universe, capable of creating new, unique Hogwarts students.
    Generate a detailed profile for a new Hogwarts student based on the user's preferences.
    Include the following sections:
    1.  **Name:** Invent a suitable name (First and Last).
    2.  **Hogwarts House:** Assign a house (Gryffindor, Hufflepuff, Ravenclaw, or Slytherin) and a brief justification for why they fit there.
    3.  **Year:** Assign them a year (e.g., First Year, Fifth Year).
    4.  **Appearance:** A concise description.
    5.  **Wand:** Describe their wand (wood, core, length, flexibility).
    6.  **Personality:** A paragraph detailing their key traits and a few quirks.
    7.  **Best Class(es):** Which classes do they excel in and why?
    8.  **Worst Class(es):** Which classes do they struggle with and why?
    9.  **Favorite Spell/Potion:** A favorite spell or a potion they enjoy.
    10. **Brief Backstory/Aspiration:** A short, imaginative background story or what they hope to achieve at Hogwarts.

    Ensure the character feels authentic to the Harry Potter world.
    """),
    ("human", "Generate a new Hogwarts character based on these preferences: {preferences}")
])
character_generator_chain = (
    {"preferences": RunnablePassthrough()}
    | character_prompt
    | llm
    | StrOutputParser()
)


# 4. Spell/Potion Creator Chain
spell_potion_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a master of magical creation in the Harry Potter universe, capable of inventing new spells and potions.
    Generate a detailed description for a new spell or potion based on the user's desired effect or name.
    If it's a spell, include:
    1.  **Name/Incantation:** Invent a Latin-sounding incantation.
    2.  **Wand Movement:** Describe the necessary wand movement.
    3.  **Effect:** Detail what the spell does.
    4.  **Difficulty:** How hard is it to cast? (e.g., Easy, Moderate, Advanced).
    5.  **Potential Side Effects/Limitations:** Any unintended consequences or specific conditions.

    If it's a potion, include:
    1.  **Name:** Invent a suitable name.
    2.  **Ingredients:** List 3-5 magical ingredients (consistent with HP lore if possible).
    3.  **Brewing Instructions:** Simple, step-by-step instructions.
    4.  **Appearance:** How does the potion look?
    5.  **Effect:** Detail what the potion does and its duration.
    6.  **Antidote/Counter-Effect:** How can its effects be reversed or countered?

    Be imaginative but ensure the creation fits within the logic and style of the Harry Potter world.
    """),
    ("human", "Invent a new spell or potion based on this idea: {idea}")
])
spell_potion_creator_chain = (
    {"idea": RunnablePassthrough()}
    | spell_potion_prompt
    | llm
    | StrOutputParser()
)


# 5. Muggle Life Profile Generator Chain (20 details)
muggle_life_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a highly creative AI, capable of reimagining Harry Potter characters as if they lived
    in our contemporary Muggle world, with no magic whatsoever.
    Generate a detailed "Muggle Life Profile" for the specified Harry Potter character.
    Focus on how their core personality traits, talents, and relationships would manifest without magic.

    Include all 20 of the following sections in your profile. Be creative, specific, and ensure consistency
    with the character's essence, even in a non-magical context:

    1.  **Character Name:** The Harry Potter character's original name.
    2.  **Muggle Profession/Career (Adult Life):** What real-world job best suits their inherent traits, non-magical skills, and long-term ambitions? Provide a brief justification.
    3.  **Education Background:** What kind of schooling or higher education would they have pursued?
    4.  **Birthplace/Nationality:** Where in the Muggle world would they likely be born and grow up, considering their original background and personality?
    5.  **Languages Spoken:** Which languages would they learn or speak fluently, besides their native tongue?
    6.  **Key Personality Traits (Muggle Context):** How would their bravery, loyalty, ambition, intelligence, eccentricity, or other defining traits manifest in everyday, non-magical situations?
    7.  **Hobbies & Interests:** What non-magical activities would they enjoy in their free time?
    8.  **Favorite Muggle Technology/Gadget:** What modern tech would fascinate them or become their go-to gadget?
    9.  **Favorite Music Genre & Artist(s):** What kind of music would they enjoy, reflecting their personality? Name a plausible genre and an artist.
    10. **Favorite Movies/TV Shows:** What types of films or specific movies/shows would resonate with them?
    11. **Favorite Food & Drink (Muggle):** What would be their go-to comfort food and beverage?
    12. **Relationship Status/Significant Other:** What kind of relationship would they seek, or who might be a plausible partner (real or fictional Muggle person, or a type of person)?
    13. **Pet (Muggle Version):** What kind of non-magical pet would they own, if any?
    14. **Fashion Style:** Describe their typical clothing style in the Muggle world.
    15. **Biggest Fear (Muggle Version):** What non-magical fear might they have, reflecting their deeper vulnerabilities?
    16. **Hidden/Unexpected Talent:** A surprising non-magical skill or talent they might possess.
    17. **Dream Home (Muggle):** Describe their ideal non-magical living space (e.g., city apartment, cozy cottage, sprawling estate).
    18. **One Quirky Habit:** A small, unique, non-magical habit or mannerism they'd have.
    19. **Social Circle:** What kind of people would make up their close friends or social group?
    20. **Aspiration/Life Goal:** What would be their primary ambition or guiding principle in their Muggle life?

    Make sure the profile is engaging, detailed, and feels like a genuine interpretation of the character in a Muggle setting.
    """),
    ("human", "Generate a Muggle Life Profile for: {character_name}")
])
muggle_life_generator_chain = (
    {"character_name": RunnablePassthrough()}
    | muggle_life_prompt
    | llm
    | StrOutputParser()
)


# 6. Image Generation Chains (Using DreamStudio API)

# 1. Chain to generate a detailed text prompt for the image model
image_prompt_generator_chain_text = ChatPromptTemplate.from_messages([
    ("system", """You are a highly creative visual describer.
    Given a short request, expand it into a detailed, high-quality, and vivid
    textual prompt for an advanced image generation AI. Focus on visual elements,
    style, lighting, and atmosphere. Ensure the description is concise enough for an image model,
    but rich in detail. Max 150 words.

    Examples:
    - User: "Harry Potter in Quidditch robes"
      AI: "A hyperrealistic image of Harry Potter in detailed red and gold Quidditch robes, flying on a broomstick above a lush green Quidditch pitch, intense focus on his face, golden snitch blurred in the foreground, dynamic lighting, cinematic quality, high resolution."
    - User: "A wand for a brave Gryffindor"
      AI: "A close-up, high-detail image of an elegant wand made from sturdy oak wood with a subtle lion's head carving near the handle, its core indicated by a faint golden glow. The background is a blurred Gryffindor common room. Magical, realistic lighting, highly detailed."

    Your response should be *only* the detailed prompt text, ready for an image model.
    """),
    ("human", "{user_request}")
])

image_prompt_generator_chain = (
    {"user_request": RunnablePassthrough()}
    | image_prompt_generator_chain_text
    | llm # Use our existing Gemini-Flash LLM for prompt generation
    | StrOutputParser()
)

# Function to call the DreamStudio API for image generation
async def generate_image_api(detailed_prompt: str):
    DREAMSTUDIO_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image" # Stable Diffusion 1.6
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {os.getenv('DREAMSTUDIO_API_KEY')}" # Using DreamStudio key
    }

    payload = {
        "text_prompts": [{"text": detailed_prompt}],
        "cfg_scale": 7,  # Classifier Free Guidance scale - higher means more adherence to prompt
        "clip_guidance_preset": "FAST_BLUE", # Optimization for speed/quality
        "height": 512,   # Image height (common for SD 1.x models)
        "width": 512,    # Image width
        "samples": 1,    # Number of images to generate (1 for free tier)
        "steps": 30      # Number of inference steps
    }

    try:
        response = requests.post(DREAMSTUDIO_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()

        if response_data.get("artifacts"):
            for i, image_artifact in enumerate(response_data["artifacts"]):
                if image_artifact.get("base64"):
                    image_base64 = image_artifact["base64"]
                    return f"data:image/png;base64,{image_base64}"
            raise Exception("No image artifact found in DreamStudio response.")
        else:
            raise Exception(f"Invalid DreamStudio response structure: {json.dumps(response_data)}")

    except requests.exceptions.RequestException as req_e:
        error_message = f"DreamStudio API Request failed: {req_e.response.status_code if req_e.response else 'N/A'} {req_e.response.reason if req_e.response else 'N/A'}"
        if req_e.response is not None:
            error_message += f"\nResponse body: {req_e.response.text}"
        raise Exception(error_message)
    except Exception as e:
        raise Exception(f"Image generation failed: {e}. Ensure DREAMSTUDIO_API_KEY is correct and billing/credits are available on DreamStudio.")


# --- Streamlit UI Layout and Logic ---
# Keep layout centered as per original
st.set_page_config(page_title="Harry Potter GenAI Assistant", layout="centered")

# Updated title with wand emoji
st.title("🪄 Harry Potter GenAI Assistant")
st.markdown("Ask questions, generate stories/characters, create spells, explore Muggle lives, or generate images!")

# --- Custom CSS for Simple Pulsating Wand in Spinner (only this CSS) ---
st.markdown(
    """
    <style>
    /* Keyframes for the pulsation animation */
    @keyframes pulsate-wand {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }

    /* Apply animation to elements with this class */
    .magic-pulsate {
        display: inline-block; /* Ensure transform applies correctly */
        animation: pulsate-wand 1.5s infinite ease-in-out;
        margin-right: 5px; /* Small space after wand */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- End Custom CSS ---

# Updated feature choice to include all 6 options
feature_choice = st.radio(
    "Choose a feature:",
    ("Ask a Question (RAG)",
     "Generate a 'What If' Story",
     "Generate a Hogwarts Character",
     "Invent a Spell/Potion",
     "Generate a Muggle Life Profile",
     "Generate an Image"
    ),
    horizontal=False, # Vertical layout for many options
    key="feature_selector"
)

# Initialize session state for image generation
if 'generated_image_url' not in st.session_state:
    st.session_state.generated_image_url = None

# Conditional UI for Image Generation
if feature_choice == "Generate an Image":
    st.markdown("Describe the image you want to generate (e.g., 'Harry's wand', 'Dobby in a new outfit', 'Hogwarts castle in winter').")
    image_input = st.text_area(
        "Enter image description:",
        placeholder="e.g., Luna Lovegood with her Spectrespecs, a new magical creature, a potion bottle with glowing liquid",
        height=100,
        key="image_description_input"
    )

    if st.button("Generate Image", key="generate_image_button"):
        if image_input:
            # Use st.spinner directly with a custom text containing the animated wand
            with st.spinner("Conjuring your image... <span class='magic-pulsate'>🪄</span> This can take 15-30 seconds or more!"):
                try:
                    # Step 1: Generate detailed prompt from user input using Gemini
                    detailed_prompt = image_prompt_generator_chain.invoke({"user_request": image_input})
                    st.info(f"Generating image with prompt: '{detailed_prompt}'") # Show prompt for debugging

                    # Step 2: Call DreamStudio Image Generation API
                    generated_image_url = asyncio.run(generate_image_api(detailed_prompt))
                    st.session_state.generated_image_url = generated_image_url
                    st.success("Image generated!")
                except Exception as e:
                    st.error(f"Failed to generate image: {e}")
                    st.session_state.generated_image_url = None
                finally:
                    pass # No explicit stop needed as spinner handles its own removal

        else:
            st.warning("Please enter a description for the image.")

    if st.session_state.generated_image_url:
        st.image(st.session_state.generated_image_url, caption="Generated Image", use_container_width=True) # Changed to use_container_width
        st.download_button(
            label="Download Image",
            data=base64.b64decode(st.session_state.generated_image_url.split(",")[1]),
            file_name="generated_hp_image.png",
            mime="image/png"
        )
else: # Default UI for other features
    # Clear image state if user switches away from the image tab
    if st.session_state.generated_image_url:
        st.session_state.generated_image_url = None

    user_input_placeholder = ""
    if feature_choice == "Ask a Question (RAG)":
        user_input_placeholder = "e.g., Who is Dobby and what is his role in the books?"
    elif feature_choice == "Generate a 'What If' Story":
        user_input_placeholder = "e.g., What if Snape didn't die?"
    elif feature_choice == "Generate a Hogwarts Character":
        user_input_placeholder = "e.g., A brave Gryffindor who loves magical creatures, or A cunning Slytherin obsessed with ancient runes."
    elif feature_choice == "Invent a Spell/Potion":
        user_input_placeholder = "e.g., A spell to summon a cup of tea, or a potion for temporary invisibility."
    elif feature_choice == "Generate a Muggle Life Profile":
        user_input_placeholder = "e.g., Harry Potter, or Hermione Granger, or Luna Lovegood."

    user_input = st.text_area(
        f"Enter your {feature_choice.split('(')[0].strip().lower()} details:",
        placeholder=user_input_placeholder,
        height=150,
        key="user_query_input"
    )

    if st.button("Generate Response", key="generate_button", help="Click to get an answer or story."):
        if user_input:
            # Use st.spinner directly with a custom text containing the animated wand
            with st.spinner("Processing your request... <span class='magic-pulsate'>🪄</span> This may take a moment."):
                try:
                    if feature_choice == "Ask a Question (RAG)":
                        response = rag_chain.invoke({"input": user_input})
                        st.subheader("Answer:")
                        st.write(response["answer"])

                        st.subheader("Retrieved Context Snippets:")
                        if response["context"]:
                            for i, doc in enumerate(response["context"]):
                                source_filename = os.path.basename(doc.metadata.get('source', 'Unknown Source'))
                                st.markdown(f"**From: {source_filename}**")
                                st.write(doc.page_content[:500] + "...")
                                st.markdown("---")
                        else:
                            st.info("No specific context was retrieved for this question. The answer might be general knowledge.")

                    elif feature_choice == "Generate a 'What If' Story":
                        generated_story = story_generator_chain.invoke(user_input)
                        st.subheader("Generated Story:")
                        st.write(generated_story)
                        st.info("This is a creative generation. While grounded in character knowledge from the books, the plot is speculative!")

                    elif feature_choice == "Generate a Hogwarts Character":
                        generated_character = character_generator_chain.invoke(user_input)
                        st.subheader("Generated Hogwarts Character Profile:")
                        st.write(generated_character)
                        st.info("This character is newly generated based on your preferences and the Harry Potter universe lore!")

                    elif feature_choice == "Invent a Spell/Potion":
                        generated_creation = spell_potion_creator_chain.invoke(user_input)
                        st.subheader("Generated Spell/Potion Details:")
                        st.write(generated_creation)
                        st.info("This is a creative generation. Effects are fictional and for entertainment purposes only!")

                    elif feature_choice == "Generate a Muggle Life Profile":
                        generated_profile = muggle_life_generator_chain.invoke(user_input)
                        st.subheader("Generated Muggle Life Profile:")
                        st.write(generated_profile)
                        st.info("This profile is a creative interpretation of the character in a non-magical setting!")

                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
                    st.warning("Please ensure your Google API Key is correctly configured in your .env file.")
        else:
            st.warning("Please enter your query in the text area to get a response.")

st.markdown("---")
st.caption("Powered by LangChain, Google Gemini, and your Harry Potter books.")
st.caption("Contact at siri12091713@gmail.com")