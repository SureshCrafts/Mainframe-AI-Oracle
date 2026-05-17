import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_classic.chains import RetrievalQA

# --- Page Configuration ---
st.set_page_config(page_title="Fiserv Tech Mentor", page_icon="💳", layout="wide")

# --- Custom CSS for Orange Text on Black Background ---
st.markdown("""
    <style>
    /* Global Background and Text Color */
    .stApp, div[data-testid="stToolbar"] {
        background-color: #000000 !important;
    }
    
    /* Force ALL text to be Fiserv Orange */
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6, p, label, span, li {
        color: #ff6600 !important;
        font-family: 'Courier New', Courier, monospace; /* Monospace feels like a Mainframe terminal */
    }

    /* Style the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #ff6600;
    }

    /* Style Input Boxes (Background dark, text orange) */
    input {
        background-color: #000000 !important;
        color: #ff6600 !important;
        border: 1px solid #ff6600 !important;
    }

    /* Style the Divider */
    hr {
        border-top: 1px solid #ff6600 !important;
    }

    /* Success/Info boxes text color override */
    .stAlert p {
        color: #ff6600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("FISERV MAINFRAME APP AI")
st.subheader("LEGACY TECH MENTOR")
st.divider()

# --- Sidebar for Personal/Professional Context ---
with st.sidebar:
    st.header("DEVELOPER PROFILE")
    st.markdown("**NAME:** SURESH KRISHNAMURTHY")
    st.markdown("**ROLE:** MAINFRAME DEVELOPER")
    st.markdown("**DOMAIN:** CARDS & BANKING (FISERV)")
    st.divider()
    api_key = st.text_input("ENTER OPENAI API KEY", type="password")

if api_key:
    try:
        os.environ["OPENAI_API_KEY"] = api_key

        if 'vectorstore' not in st.session_state:
            with st.spinner("INDEXING KNOWLEDGE BASE..."):
                # Using TextLoader to ensure .cbl and .jcl files are read as plain text
                loader = DirectoryLoader(
                    './knowledge_base', 
                    glob="**/*.*", 
                    loader_cls=TextLoader
                )
                docs = loader.load()
                embeddings = OpenAIEmbeddings()
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever()
        )

        # --- Interaction ---
        user_question = st.text_input("ENTER QUERY (E.G. EXPLAIN LATE FEE GRACE PERIOD):")

        if user_question:
            with st.spinner("ANALYZING SOURCE CODE..."):
                response = qa.run(user_question)
                st.markdown("### ANALYSIS OUTPUT")
                st.write(response)

    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")
else:
    st.warning("PLEASE PROVIDE AUTHORIZATION KEY TO PROCEED.")