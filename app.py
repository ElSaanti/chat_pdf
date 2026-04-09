import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# App title and presentation
st.title('Asistente de Estudio Inteligente 📚💬')
st.write("Una herramienta pensada para ayudarte a entender textos largos después de un día cansado 😌")
st.write("Versión de Python:", platform.python_version())

# Load and display image
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar information
with st.sidebar:
    st.subheader("💡 Sobre esta app")
    st.write("Este asistente te ayuda a resumir, entender y responder preguntas sobre documentos PDF de forma sencilla.")

# Get API key from user
st.subheader("🔑 Configuración inicial")
ke = st.text_input('Ingresa tu clave de OpenAI', type="password")

# Expandable instructions
with st.expander("❓ ¿No tienes una API Key? Aprende cómo obtenerla (haz clic aquí)"):
    st.markdown("""
    Sigue estos pasos para obtener tu API Key de OpenAI:

    1. Ve a: https://platform.openai.com/
    2. Crea una cuenta o inicia sesión
    3. Dirígete a "API Keys"
    4. Haz clic en "Create new secret key"
    5. Copia la clave y pégala aquí

    ⚠️ Importante:
    - No compartas tu clave con nadie
    - Puede tener costos asociados dependiendo del uso
    """)

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
st.subheader("📄 Sube tu material de estudio")
pdf = st.file_uploader("Carga un PDF (apuntes, libros, artículos...)", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Se extrajeron {len(text)} caracteres del documento")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"El documento fue dividido en {len(chunks)} partes para facilitar el análisis")
        
        # Create embeddings and knowledge base
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # User question interface
        st.subheader("🧠 Hazle preguntas a tu documento")
        st.write("Puedes pedir resúmenes, explicaciones simples o resolver dudas específicas ✨")
        user_question = st.text_area(" ", placeholder="Ej: Explícame este tema como si estuviera cansado...")

        # Process question when submitted
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            
            # Load QA chain
            chain = load_qa_chain(llm, chain_type="stuff")
            
            # Run the chain
            response = chain.run(input_documents=docs, question=user_question)
            
            # Display the response
            st.markdown("### 📌 Respuesta clara y sencilla:")
            st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Necesitas ingresar tu API Key antes de continuar")
else:
    st.info("👆 Sube un PDF para empezar a estudiar de forma más fácil")
