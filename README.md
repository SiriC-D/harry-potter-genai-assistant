# Harry Potter GenAI Assistant 🪄

A Streamlit application that acts as a magical AI assistant for the Harry Potter universe.
It can answer questions based on the Harry Potter books, generate "what if" stories,
create new Hogwarts characters, invent spells/potions, generate Muggle life profiles
for characters, and even generate images based on HP themes!

## Features ✨

-   **RAG (Retrieval-Augmented Generation):** Answer questions using knowledge extracted from Harry Potter books.
-   **"What If" Story Generation:** Explore imaginative scenarios within the HP universe.
-   **Hogwarts Character Generator:** Create detailed profiles for new students.
-   **Spell/Potion Creator:** Invent new magical items with descriptions.
-   **Muggle Life Profile:** Reimagine HP characters in a non-magical, contemporary world.
-   **Image Generation:** Generate thematic images based on user prompts (requires DreamStudio API key).

## Setup and Installation 🧹

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SiriC-D/harry-potter-genai-assistant.git](https://github.com/SiriC-D/harry-potter-genai-assistant.git)
    cd harry-potter-genai-assistant
    ```
2.  **Create a Python virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare your Harry Potter Books:**
    Place your `.txt` files of the Harry Potter books into a folder named `harry_potter_books/` in the root of the project (the same place as `app.py`). The app expects these files to be in place for the RAG functionality.

5.  **Set up API Keys:**
    * Obtain a **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
    * Obtain a **DreamStudio API Key** from [DreamStudio](https://platform.stability.ai/account/keys). (Optional, only needed for image generation)
    * Create a file named `.env` in the root of your project (the same directory as `app.py`) and add your keys like this:
        ```
        GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        DREAMSTUDIO_API_KEY="YOUR_DREAMSTUDIO_API_KEY_HERE"
        ```
        **Important:** Do NOT commit your `.env` file to GitHub! It is already listed in your `.gitignore` file.

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your web browser. The first time you run it, it will create the `faiss_index_hp` vector store (this might take a few minutes depending on the size of your books).

## Usage 🧙‍♀️

Select a feature from the radio buttons and provide the necessary input. The AI will then generate a response or an image!

## Contact 📧

For any questions or feedback, please contact me at sirichandana20082006@gmail.com