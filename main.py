import os
import io
import re
import fitz  # PyMuPDF
import google.generativeai as genai
from google.generativeai import types as genai_types  # For embedding config
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal, Generator, Union
from dotenv import load_dotenv
from PIL import Image
import json
import shutil
import uuid
from pathlib import Path
import traceback
import pandas as pd
import chromadb  # Import chromadb

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ============================
# CONFIGURATION & SETUP
# ============================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TEMP_STORAGE_PATH = Path(os.getenv("TEMP_STORAGE_PATH", "./temp_files"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db"))
MAX_CONTEXT_CHARS = 200000  # Approx 50k tokens
MAX_CHAT_PREVIEW_CHARS = 2000  # Preview for text files in chat

if not GOOGLE_API_KEY:
    print("⚠️ Warning: GOOGLE_API_KEY not found.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

TEMP_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

# --- ChromaDB Client ---
chroma_client = None
try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    print(f"ChromaDB client initialized. Path: {CHROMA_DB_PATH}")
except Exception as chroma_err:
    print(f"🚨 Failed to initialize ChromaDB client: {chroma_err}")

# ============================
# PYDANTIC MODELS
# ============================

class ChatHistoryItem(BaseModel):
    is_user: bool
    content: str
    role: Optional[Literal['system', 'human', 'ai']] = None


class ChatRequest(BaseModel):
    query: str
    history: List[ChatHistoryItem] = []
    file_identifier: Optional[str] = None  # Optional for general conversation
    file_type: Optional[str] = None        # Optional for general conversation
    analysis_mode: Optional[str] = "insights"  # Default analysis mode


class Insight(BaseModel):
    id: str
    text: str
    category: str
    sources: List[str] = []
    confidence: float
    tags: List[str] = []


class VisualizationSuggestion(BaseModel):
    id: Optional[str] = None
    type: str
    title: str
    description: str


class AnalyzeRequest(BaseModel):
    file_identifier: str
    file_type: str
    analysis_type: Literal["insights", "visualizations", "full"] = "full"


class AnalyzeResponse(BaseModel):
    insights: List[Insight] = []
    suggested_visualizations: List[VisualizationSuggestion] = []
    parsing_metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ReportSectionModel(BaseModel):
    id: str
    title: str
    content: str
    type: Literal['introduction', 'main', 'analysis', 'conclusion', 'references']


class ReportRequest(BaseModel):
    file_identifier: str
    file_type: str
    report_format: Literal['detailed', 'summary', 'presentation'] = 'detailed'
    insights: Optional[List[Insight]] = None
    visualizations: Optional[List[VisualizationSuggestion]] = None


class ReportResponse(BaseModel):
    title: str
    summary: str
    sections: List[ReportSectionModel]
    error: Optional[str] = None

# ============================
# AUTHENTICATION HELPERS
# ============================

async def verify_token(authorization: Optional[str] = Header(None)):
    """Simple authentication placeholder."""
    return {"user_id": "simulated-user-123"}

# ============================
# GENERAL HELPERS
# ============================

async def save_uploaded_file(file: UploadFile) -> tuple[Path, str]:
    """Saves uploaded file temporarily and returns its path and detected content type."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
    
    sanitized_filename = re.sub(r'\s+', '_', file.filename)
    sanitized_filename = re.sub(r'[^\w\.\-]', '', sanitized_filename)
    
    if len(sanitized_filename) > 100:
        sanitized_filename = sanitized_filename[-100:]
    
    temp_filename = f"{uuid.uuid4()}_{sanitized_filename}"
    temp_filepath = TEMP_STORAGE_PATH / temp_filename
    content_type = file.content_type or 'application/octet-stream'
    
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved: {temp_filepath}, Type: {content_type}")
        return temp_filepath, content_type
    except Exception as e:
        print(f"Error saving file {file.filename}: {e}")
        print(traceback.format_exc())
        if temp_filepath.exists():
            try:
                temp_filepath.unlink(missing_ok=True)
                print(f"Cleaned up temporary file: {temp_filepath}")
            except OSError as unlink_err:
                print(f"Warning: Failed to clean up temp file {temp_filepath}: {unlink_err}")
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        await file.close()


async def get_file_path(file_identifier: str) -> Path:
    """Get the local path for a file identifier, with security checks."""
    if ".." in file_identifier or "/" in file_identifier or "\\" in file_identifier:
        raise HTTPException(status_code=400, detail="Invalid file identifier.")
    
    local_path = (TEMP_STORAGE_PATH / file_identifier).resolve()
    
    if not local_path.is_file() or not str(local_path).startswith(str(TEMP_STORAGE_PATH.resolve())):
        raise HTTPException(status_code=404, detail=f"File not found: {file_identifier}")
    
    return local_path


def get_gemini_model(model_name: str = "gemini-2.0-flash", **kwargs):
    """Get a Gemini generative model with configured settings."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set for Gemini")
    
    safety_settings = {}
    generation_config = genai_types.GenerationConfig(
        max_output_tokens=16384,
        temperature=1.0
    )
    
    return genai.GenerativeModel(
        model_name=model_name,
        safety_settings=safety_settings,
        generation_config=generation_config,
        **kwargs
    )


def get_langchain_chat_model(model_name: str = "gemini-2.0-flash", **kwargs):
    """Get a LangChain Gemini chat model with configured settings."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set for LangChain")
    
    final_kwargs = {'temperature': 1.0, **kwargs}
    if 'stream' in final_kwargs:
        del final_kwargs['stream']
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        **final_kwargs
    )


def convert_chat_history_to_langchain(history: List[ChatHistoryItem]) -> List[BaseMessage]:
    """Converts frontend chat history to LangChain message format."""
    messages: List[BaseMessage] = []
    
    for item in history:
        role = item.role or ('human' if item.is_user else 'ai')
        msg_content = item.content if item.content is not None else ""
        
        if role == 'system':
            # Instead of SystemMessage, use HumanMessage with a prefix
            messages.append(HumanMessage(content=f"[System]: {msg_content}"))
        elif role == 'human':
            messages.append(HumanMessage(content=msg_content))
        elif role == 'ai':
            messages.append(AIMessage(content=msg_content))
    
    return messages


async def stream_langchain_response(model: ChatGoogleGenerativeAI, messages: List[BaseMessage]) -> Generator[str, Any, None]:
    """Streams response from a LangChain Chat Model, yielding content chunks."""
    try:
        async for chunk in model.astream(messages):
            yield str(chunk.content)
    except Exception as e:
        print(f"LC Stream Error: {e}")
        yield f"\n\n[LLM Error: {e}]"  # Yield error message within the stream


async def stream_agent_response(agent, user_input: str) -> Generator[str, Any, None]:
    """Streams response from a LangChain agent, handling output formatting."""
    full_output = ""
    try:
        async for chunk in agent.astream({"input": user_input}):
            if isinstance(chunk, dict) and "output" in chunk and chunk["output"] is not None:
                new_output = str(chunk["output"])
                if new_output.startswith(full_output):
                    yield new_output[len(full_output):]
                else:
                    yield new_output
                full_output = new_output
    except Exception as e:
        print(f"Agent Stream Error: {e}")
        yield f"\n\n[Agent Error: {e}]"


def convert_dtype_to_str(dtype) -> str:
    """Converts pandas data types to string descriptions for prompts."""
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    elif pd.api.types.is_float_dtype(dtype):
        return "float"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_categorical_dtype(dtype):
        return "categorical"
    return "object"  # Default for strings and other types

# ============================
# RAG-SPECIFIC HELPERS
# ============================

OCR_CHUNK_PROMPT = """\
OCR the following document page image into Markdown.
Chunk the text into meaningful sections of roughly 250-750 words based on semantic breaks (paragraphs, headings).
Surround each chunk with <chunk> and </chunk> tags. Ensure chunks are meaningful and complete.

- Format tables STRICTLY as simple Markdown tables.
- For figures/charts: Insert a placeholder like `[Figure: Brief description of figure content]`.

Output ONLY the processed Markdown with chunk tags and specified placeholders. No explanations before or after.
"""


def extract_chunks_from_ocr(ocr_text: str) -> List[str]:
    """Extract chunks from OCR text using chunk tags or fallback to paragraphs."""
    chunks = re.findall(r"<chunk>(.*?)</chunk>", ocr_text, re.DOTALL)
    if not chunks:
        chunks = [c.strip() for c in ocr_text.split("\n\n") if c.strip()]
    return [c.strip() for c in chunks if c.strip()]


async def process_pdf_for_rag(local_path: Path, collection_name: str) -> tuple[int, Optional[str]]:
    """Process a PDF file for RAG, extracting text via OCR and storing in ChromaDB."""
    if not chroma_client:
        return 0, "ChromaDB client not initialized."
    
    if not GOOGLE_API_KEY:
        return 0, "Google API Key not set for processing."
    
    all_chunks_data = []
    page_count = 0
    ocr_model = get_gemini_model("gemini-2.0-flash")
    
    try:
        pdf_document = fitz.open(str(local_path))
        page_count = pdf_document.page_count
        
        for i in range(page_count):
            page = pdf_document[i]
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                response = await ocr_model.generate_content_async([OCR_CHUNK_PROMPT, img])
                page_chunks = extract_chunks_from_ocr(response.text)
                for idx, chunk_text in enumerate(page_chunks):
                    all_chunks_data.append((f"page_{i+1}_chunk_{idx+1}", chunk_text))
            except Exception as page_err:
                print(f"⚠️ Warning: Error processing page {i+1}: {page_err}")
        
        pdf_document.close()
        
        if not all_chunks_data:
            return page_count, "No text chunks could be extracted."
        
        chunk_ids = [item[0] for item in all_chunks_data]
        chunk_texts = [item[1] for item in all_chunks_data]
        
        embedding_response = genai.embed_content(
            model="models/text-embedding-004",
            content=chunk_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = [e for e in embedding_response['embedding']]
        
        if len(embeddings) != len(all_chunks_data):
            return page_count, f"Mismatch chunks/embeddings."
        
        collection = chroma_client.get_or_create_collection(name=collection_name)
        metadatas = [{"text": text} for text in chunk_texts]
        collection.upsert(ids=chunk_ids, embeddings=embeddings, metadatas=metadatas)
        
        print(f"Upserted {len(chunk_ids)} chunks into Chroma '{collection_name}'.")
        return page_count, None
    except Exception as e:
        error_msg = f"PDF RAG processing failed: {e}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        return page_count, error_msg

# ============================
# PROMPT TEMPLATES
# ============================

GENERIC_INSIGHTS_VIZ_PROMPT = """\
You are an expert analyst and data interpreter. Analyze the {file_type} content below and provide:
1. Key insights - notable patterns, findings, or interesting facts
2. Visualization suggestions - charts, graphs or visual representations that would help communicate this data

Format your response as JSON with the following structure:
```json
{{
  "insights": [
    {{
      "id": "insight-1",
      "text": "Clear description of the insight",
      "category": "One of: [General, Data Quality, Pattern, Trend, Anomaly, Relationship]",
      "sources": ["Relevant sections or data points"],
      "confidence": 0.95,
      "tags": ["relevant", "tags", "for", "categorization"]
    }}
  ],
  "suggested_visualizations": [
    {{
      "type": "One of: [bar_chart, line_chart, scatter_plot, pie_chart, heatmap, table, custom]",
      "title": "Suggested title for the visualization",
      "description": "Detailed description of what this visualization would show and why it's useful"
    }}
  ]
}}
```

CONTENT:
{context}
"""

CSV_ANALYSIS_PROMPT = """\
You are an expert data analyst. Analyze this CSV file and provide:
1. Key insights about the data
2. Visualization suggestions that would best communicate patterns and insights

File details:
- Rows: {row_count}
- Columns: {column_count}
- Headers: {column_headers}
- Data types: {data_types}

Sample data:
{sample_rows}

Format your response as JSON with the following structure:
```json
{{
  "insights": [
    {{
      "id": "insight-1",
      "text": "Clear description of the insight",
      "category": "One of: [General, Data Quality, Pattern, Trend, Anomaly, Relationship]",
      "sources": ["Relevant column names or data points"],
      "confidence": 0.95,
      "tags": ["relevant", "tags", "for", "categorization"]
    }}
  ],
  "suggested_visualizations": [
    {{
      "type": "One of: [bar_chart, line_chart, scatter_plot, pie_chart, heatmap, table, custom]",
      "title": "Suggested title for the visualization",
      "description": "Detailed description of what this visualization would show and why it's useful"
    }}
  ]
}}
```
"""

REPORT_GENERATION_PROMPT = """\
Generate a professional {report_format} report on the topic: "{topic}".

File details:
- Type: {file_type}
- Identifier: {file_identifier}

Key insights identified:
{insights_list}

Use the following content to create your report:
{context}

Structure the report with suitable sections based on the content and requested format.

Format your response as JSON with this structure:
```json
{{
  "title": "Appropriate Report Title",
  "summary": "Concise executive summary (2-4 sentences)",
  "sections": [
    {{
      "id": "sec-introduction",
      "title": "Introduction",
      "content": "Full section content with markdown formatting",
      "type": "introduction"
    }},
    {{
      "id": "sec-findings-1",
      "title": "Finding 1 Title",
      "content": "Full section content with markdown formatting",
      "type": "main"
    }},
    {{
      "id": "sec-analysis",
      "title": "Analysis",
      "content": "Full analysis section with markdown formatting",
      "type": "analysis" 
    }},
    {{
      "id": "sec-conclusion",
      "title": "Conclusion",
      "content": "Full conclusion with markdown formatting",
      "type": "conclusion"
    }},
    {{
      "id": "sec-references",
      "title": "References",
      "content": "Source citations and references",
      "type": "references"
    }}
  ]
}}
```

For {report_format} format:
- "detailed": Include comprehensive analysis with multiple sections
- "summary": Brief overview focused on key points only
- "presentation": Bulleted points suitable for slides
"""

# ============================
# FASTAPI APPLICATION
# ============================

app = FastAPI(title="Valkyry AI Service (RAG + Agent + General Chat)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.post("/api/v1/upload", response_model=Dict[str, str])
async def upload_file_endpoint(file: UploadFile = File(...), auth: dict = Depends(verify_token)):
    """Upload a file and get a file identifier."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    
    try:
        temp_filepath, content_type = await save_uploaded_file(file)
        return {"file_identifier": temp_filepath.name, "file_type": content_type}
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Upload Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest, auth: dict = Depends(verify_token)):
    """
    Handles streaming chat requests.
    - Uses RAG for PDFs
    - Uses Pandas Agent for CSVs
    - Uses standard LLM for other file types and normal conversations
    """
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="AI Service not configured.")

    # Check if this is a file-based or normal conversation
    use_file_context = request.file_identifier and request.file_type
    is_pdf = use_file_context and request.file_type == "application/pdf"
    is_csv = use_file_context and request.file_type == "text/csv"

    # Check if ChromaDB is available for PDF processing
    if is_pdf and not chroma_client:
        async def error_stream_chroma():
            yield f"data: {json.dumps({'error': 'Vector DB not ready.'})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        return StreamingResponse(error_stream_chroma(), media_type="text/event-stream")

    # Initialize LLM
    try:
        llm = get_langchain_chat_model()
    except Exception as llm_init_err:
        async def error_stream_llm():
            yield f"data: {json.dumps({'error': f'AI model init failed: {llm_init_err}'})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
        return StreamingResponse(error_stream_llm(), media_type="text/event-stream")

    # Convert chat history to LangChain format
    langchain_history = convert_chat_history_to_langchain(request.history)

    async def stream_response_generator():
        try:
            if is_pdf:
                # --- RAG for PDF ---
                print(f"Using RAG for PDF: {request.file_identifier}")
                collection_name = request.file_identifier
                
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    query_emb_resp = genai.embed_content(
                        model="models/text-embedding-004",
                        content=[request.query],
                        task_type="RETRIEVAL_QUERY"
                    )
                    query_embedding = query_emb_resp['embedding']
                    results = collection.query(query_embeddings=query_embedding, n_results=5)
                    
                    if not results or not results.get("ids") or not results["ids"][0]:
                        yield f"data: {json.dumps({'token': '[System: No relevant info found.]'})}\n\n"
                    else:
                        context = "\n\n".join([m['text'] for m in results['metadatas'][0]])
                        rag_prompt = f"Use context ONLY...\nContext:\n---\n{context}\n---\nQuestion: {request.query}\nAnswer:"
                        messages_for_llm = [
    # Instead of SystemMessage, use HumanMessage with a prefix
    HumanMessage(content="[System]: Answer based ONLY on context.\n\n" + rag_prompt)
]
                        async for token in stream_langchain_response(llm, messages_for_llm):
                            yield f"data: {json.dumps({'token': token})}\n\n"
                
                except Exception as rag_err:
                    error_detail = f"PDF RAG Error: {rag_err}"
                    print(f"{error_detail}\n{traceback.format_exc()}")
                    
                    if "does not exist" in str(rag_err).lower():
                        yield f"data: {json.dumps({'token': '[System: Analyze first.]'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': str(error_detail)})}\n\n"
            
            elif is_csv:
                # --- Pandas Agent for CSV ---
                print(f"Using Pandas Agent for CSV: {request.file_identifier}")
                
                try:
                    local_path = await get_file_path(request.file_identifier)
                    df = pd.read_csv(local_path, low_memory=False)
                    df.columns = ["_".join(str(col).split()).lower() for col in df.columns]
                    
                    agent = create_pandas_dataframe_agent(
                        llm, 
                        df, 
                        verbose=False, 
                        agent_executor_kwargs={"handle_parsing_errors": True}, 
                        return_intermediate_steps=False
                    )
                    
                    async for chunk in stream_agent_response(agent, request.query):
                        yield f"data: {json.dumps({'token': chunk})}\n\n"
                
                except pd.errors.EmptyDataError:
                    yield f"data: {json.dumps({'token': '[System: CSV empty.]'})}\n\n"
                except Exception as agent_err:
                    error_detail = f"Pandas Agent Error: {agent_err}"
                    print(f"{error_detail}\n{traceback.format_exc()}")
                    yield f"data: {json.dumps({'error': str(error_detail)})}\n\n"
            
            else:
                # --- Standard LLM Chat ---
                system_prompt_content = "You are a helpful AI assistant."
                
                if use_file_context:
                    try:
                        local_path = await get_file_path(request.file_identifier)
                        print(f"Std chat for file: {request.file_identifier}")
                        context_summary = f"File: '{local_path.name}' ({request.file_type})."
                        
                        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                            preview = f.read(MAX_CHAT_PREVIEW_CHARS)
                            context_summary += f"\nPreview:\n{preview}{'...' if len(preview)==MAX_CHAT_PREVIEW_CHARS else ''}"
                        
                        system_prompt_content = f"Assistant answering about a file. {context_summary}"
                    except Exception as ctx_err:
                        print(f"Warn: Ctx build failed: {ctx_err}")
                        system_prompt_content = f"Assistant answering about file '{request.file_identifier}'. Preview unavailable."
                else:
                    print("Std chat without file context.")
                
                messages_for_llm = [
    # Instead of SystemMessage, use HumanMessage with a prefix
    HumanMessage(content=f"[System]: {system_prompt_content}\n\n{request.query}")
]
                async for token in stream_langchain_response(llm, messages_for_llm):
                    yield f"data: {json.dumps({'token': token})}\n\n"
        
        except Exception as stream_gen_err:
            error_detail = f"Stream Gen Error: {stream_gen_err}"
            print(f"{error_detail}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': str(error_detail)})}\n\n"
        
        finally:
            yield f"data: {json.dumps({'status': 'done'})}\n\n"
    
    return StreamingResponse(stream_response_generator(), media_type="text/event-stream")


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest, auth: dict = Depends(verify_token)):
    """Analyzes a file to extract insights and visualization suggestions."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="AI key missing.")
    
    analysis_context = ""
    prompt_to_use = GENERIC_INSIGHTS_VIZ_PROMPT
    prompt_format_args = {}
    parsing_metadata = {"loader_used": "unknown"}
    error_message = None
    insights = []
    suggestions = []
    page_count = 0
    
    try:
        local_path = await get_file_path(request.file_identifier)
        
        if request.file_type == "application/pdf":
            parsing_metadata["loader_used"] = "gemini_vision_rag"
            collection_name = request.file_identifier
            page_count, rag_error = await process_pdf_for_rag(local_path, collection_name)
            parsing_metadata["total_pages"] = page_count
            
            if rag_error:
                error_message = f"PDF RAG setup failed: {rag_error}"
                try:
                    pdf_doc = fitz.open(str(local_path))
                    analysis_context = "\n".join([page.get_text("text", sort=True) for page in pdf_doc])[:MAX_CONTEXT_CHARS]
                    pdf_doc.close()
                    parsing_metadata["loader_used"] += "_fallback_pymupdf"
                    prompt_format_args = {"file_type": "pdf (fallback)", "context": analysis_context}
                    print("Used fallback.")
                except Exception as fb_err:
                    analysis_context = ""
                    error_message += f". Fallback failed: {fb_err}"
                    prompt_format_args = {"file_type": "pdf", "context": "[Error extracting text]"}
            else:
                try:
                    collection = chroma_client.get_collection(name=collection_name)
                    results = collection.get(limit=3, include=['metadatas'])
                    analysis_context = "\n\n".join([m['text'] for m in results['metadatas']])[:MAX_CONTEXT_CHARS]
                    prompt_format_args = {"file_type": request.file_type, "context": analysis_context}
                    print("Using RAG chunks.")
                except Exception as ctx_err:
                    analysis_context = "[RAG OK, ctx preview failed]"
                    error_message = error_message or f"RAG ctx preview failed: {ctx_err}"
                    prompt_format_args = {"file_type": request.file_type, "context": analysis_context}
        
        elif request.file_type == "text/csv":
            parsing_metadata["loader_used"] = "pandas_summary"
            try:
                df = pd.read_csv(local_path, low_memory=False)
                df.columns = ["_".join(str(col).split()).lower() for col in df.columns]
                row_count, col_count = df.shape
                headers = df.columns.tolist()
                dtypes_dict = {col: convert_dtype_to_str(dtype) for col, dtype in df.dtypes.items()}
                sample_head = df.head(3).to_string(index=False)
                sample_tail = df.tail(3).to_string(index=False, header=False)
                sample_rows = f"First 3 Rows:\n{sample_head}\n...\nLast 3 Rows:\n{sample_tail}"
                
                parsing_metadata.update({
                    "estimated_rows": row_count,
                    "estimated_cols": col_count,
                    "column_headers": headers,
                    "data_types": dtypes_dict
                })
                
                prompt_to_use = CSV_ANALYSIS_PROMPT
                prompt_format_args = {
                    "file_type": request.file_type,
                    "row_count": row_count,
                    "column_count": col_count,
                    "column_headers": ", ".join(headers),
                    "data_types": json.dumps(dtypes_dict),
                    "sample_rows": sample_rows
                }
                analysis_context = "[CSV Summary]"
            except Exception as pd_err:
                error_message = f"CSV processing error: {pd_err}"
                analysis_context = ""
                print(f"{error_message}\n{traceback.format_exc()}")
        
        elif request.file_type.startswith("text/") or request.file_type == "application/json":
            parsing_metadata["loader_used"] = "text_reader"
            try:
                with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                    analysis_context = f.read(MAX_CONTEXT_CHARS + 1000)
                
                prompt_format_args = {
                    "file_type": request.file_type,
                    "context": analysis_context[:MAX_CONTEXT_CHARS]
                }
                
                if len(analysis_context) > MAX_CONTEXT_CHARS:
                    parsing_metadata["context_truncated"] = True
            except Exception as read_err:
                error_message = f"Error reading file: {read_err}"
                analysis_context = ""
        else:
            error_message = f"Unsupported file type: {request.file_type}"
            analysis_context = ""

        if analysis_context and not error_message:
            analysis_prompt_text = prompt_to_use.format(**prompt_format_args)
            try:
                analysis_model = get_gemini_model("gemini-2.0-flash")
                print(f"Sending analysis prompt...")
                response = await analysis_model.generate_content_async(analysis_prompt_text)
                
                try:
                    if not response.parts:
                        raise ValueError(f"AI response blocked/empty. Reason: {response.prompt_feedback.block_reason}")
                    
                    json_text = response.text
                    json_text_cleaned = json_text.strip().removeprefix("```json").removesuffix("```").strip()
                    
                    if not json_text_cleaned.startswith("{") or not json_text_cleaned.endswith("}"):
                        raise ValueError(f"Invalid JSON structure. Start: {json_text_cleaned[:50]}...")
                    
                    parsed_result = json.loads(json_text_cleaned)
                    insights = [Insight(**item) for item in parsed_result.get("insights", [])]
                    suggestions = [VisualizationSuggestion(**item) for item in parsed_result.get("suggested_visualizations", [])]
                    print(f"Parsed: {len(insights)} insights, {len(suggestions)} viz.")
                
                except (json.JSONDecodeError, ValueError, Exception) as parse_err:
                    error_msg = f"Processing AI analysis response failed: {parse_err}. Raw start: {response.text[:200]}..." if hasattr(response, 'text') else f"Processing AI analysis response failed: {parse_err}"
                    if not error_message:
                        error_message = error_msg
                    print(f"Analysis Response Parsing Error: {error_message}")
            
            except Exception as ai_err:
                err_str = str(ai_err).lower()
                error_msg_prefix = "AI analysis call failed:"
                if "token" in err_str or "size" in err_str:
                    error_msg_prefix = "AI model limit likely reached:"
                
                if not error_message:
                    error_message = f"{error_msg_prefix} {ai_err}"
                print(f"{error_message}\n{traceback.format_exc()}")
        
        elif not analysis_context and not error_message:
            error_message = "No content extracted."
        
        for i, sugg in enumerate(suggestions):
            sugg.id = f"viz-{i+1}"
        
        return AnalyzeResponse(
            insights=insights,
            suggested_visualizations=suggestions,
            parsing_metadata=parsing_metadata,
            error=error_message
        )
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Analyze Endpoint Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/api/v1/generate-report", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest, auth: dict = Depends(verify_token)):
    """Generates a structured report based on file analysis."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="AI key missing.")
    
    extracted_text = ""
    error_message = None
    
    try:
        local_path = await get_file_path(request.file_identifier)
        
        try:
            # Simplified context extraction
            if request.file_type == "application/pdf":
                pdf_doc = fitz.open(str(local_path))
                extracted_text = "\n".join([page.get_text("text", sort=True) for page in pdf_doc])[:MAX_CONTEXT_CHARS]
                pdf_doc.close()
            elif request.file_type == "text/csv" or request.file_type.startswith("text/") or request.file_type == "application/json":
                with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                    extracted_text = f.read(MAX_CONTEXT_CHARS + 1000)
            else:
                error_message = f"Cannot get report context for type: {request.file_type}"
        except Exception as ctx_err:
            error_message = f"Error reading context: {ctx_err}"
            extracted_text = "(Context error)"

        if not error_message or extracted_text != "(Context error)":
            report_context = extracted_text[:MAX_CONTEXT_CHARS]
            insights_list_str = "\n".join([f"- {i.text}" for i in request.insights or []]) if request.insights else "Not provided."
            topic = local_path.stem.replace('_', ' ').replace('-', ' ').title()
            
            prompt = REPORT_GENERATION_PROMPT.format(
                report_format=request.report_format,
                file_type=request.file_type,
                topic=topic,
                insights_list=insights_list_str,
                context=report_context,
                file_identifier=local_path.name
            )
            
            try:
                report_model = get_langchain_chat_model(temperature=0.8)
                print(f"Generating report for '{topic}'...")
                response = await report_model.ainvoke([HumanMessage(content=prompt)])
                ai_response_text = response.content
                
                try:
                    # Robust JSON parsing
                    json_text = ai_response_text.strip().removeprefix("```json").removesuffix("```").strip()
                    
                    if not json_text.startswith("{") or not json_text.endswith("}"):
                        raise ValueError("Invalid JSON structure")
                    
                    parsed_report = json.loads(json_text)
                    sections = [ReportSectionModel(**sec) for sec in parsed_report.get("sections", [])]
                    
                    if not any(s.type == 'references' for s in sections):
                        sections.append(ReportSectionModel(
                            id='sec-ref-fallback',
                            title='References',
                            content=f'Based on analysis of {local_path.name}',
                            type='references'
                        ))
                    
                    print(f"Report generated.")
                    return ReportResponse(
                        title=parsed_report.get("title", f"Report on {topic}"),
                        summary=parsed_report.get("summary", "N/A"),
                        sections=sections,
                        error=None
                    )
                except (json.JSONDecodeError, ValueError, Exception) as parse_err:
                    error_message = f"Failed parsing report JSON: {parse_err}. Raw: {ai_response_text[:200]}..."
                    print(error_message)
            except Exception as ai_err:
                error_message = f"AI report call failed: {ai_err}"
                print(f"{error_message}\n{traceback.format_exc()}")

        return ReportResponse(
            title="Report Error",
            summary=error_message or "Unknown error.",
            sections=[],
            error=error_message or "Unknown error"
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Generate Report Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")


@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    key_status = "Not Set"
    chroma_status = "Failed"
    
    if GOOGLE_API_KEY:
        key_status = f"Set (ends ...{GOOGLE_API_KEY[-4:]})" if len(GOOGLE_API_KEY) > 4 else "Set"
    
    if chroma_client:
        chroma_status = "Initialized"
    
    return {
        "status": "ok",
        "google_api_key_status": key_status,
        "chromadb_status": chroma_status
    }


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use the PORT environment variable provided by Render
    PORT = int(os.getenv("PORT", "8000"))
    
    print(f"Starting Valkyry AI Service (RAG + Agent + General Chat)...")
    print(f"Listening on port: {PORT}")
    # Ensure directories exist
    os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    print(f"Temp Storage: {TEMP_STORAGE_PATH.resolve()}")
    print(f"ChromaDB Path: {CHROMA_DB_PATH.resolve()}")
    print(f"Google API Key: {'Yes' if GOOGLE_API_KEY else 'No!'}")
    print(f"ChromaDB Client: {'Initialized' if chroma_client else 'Failed!'}")
    
    # Key change: disable reload in production and use the PORT env variable
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)