"""
NeoMed Digital Healthcare - Complete API + Gradio Interface
==============================================================

تحويل جميع functions من Gradio إلى FastAPI endpoints
مع الاحتفاظ بواجهة Gradio كخيار اختياري
"""

import gradio as gr
import asyncio
import logging
from pathlib import Path
import sys
import os
import json
import uuid
from datetime import datetime
from collections import defaultdict
from typing import Optional, List, Dict, Any

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from src
try:
    from src.agent import safe_run_agent_streaming, safe_run_agent, clear_memory
    from src.data_loaders import process_uploaded_file
    from src.utils import initialize_knowledge_base
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise

# ==========================================
# FastAPI Application
# ==========================================

app = FastAPI(
    title="NeoMed Digital Healthcare API",
    description="API للروبوت الطبي الذكي - نيوميد الرقمية",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Session Management
# ==========================================

# Global variable for Gradio (legacy)
processed_docs = []

# Session storage for API
api_sessions = defaultdict(lambda: {
    "chat_history": [],
    "processed_docs": [],
    "created_at": datetime.now(),
    "last_activity": datetime.now()
})

def get_api_session(session_id: str) -> dict:
    """Get or create API session"""
    api_sessions[session_id]["last_activity"] = datetime.now()
    return api_sessions[session_id]

def generate_session_id() -> str:
    """Generate unique session ID"""
    return str(uuid.uuid4())

# ==========================================
# Pydantic Models
# ==========================================

class ChatRequest(BaseModel):
    """طلب دردشة"""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("الرسالة لا يمكن أن تكون فارغة")
        return v.strip()

class ChatResponse(BaseModel):
    """رد الدردشة"""
    response: str
    session_id: str
    timestamp: str

class FileUploadResponse(BaseModel):
    """رد رفع الملف"""
    success: bool
    message: str
    session_id: str
    documents_count: int
    filename: str
    timestamp: str

class ClearMemoryResponse(BaseModel):
    """رد مسح الذاكرة"""
    success: bool
    message: str
    session_id: Optional[str] = None
    timestamp: str

class SystemStatusResponse(BaseModel):
    """حالة النظام"""
    status: str
    knowledge_base_status: str
    active_sessions: int
    total_sessions: int
    uptime: str
    timestamp: str

# ==========================================
# Initialize Knowledge Base
# ==========================================

logger.info("Initializing knowledge base...")
try:
    knowledge_base = initialize_knowledge_base()
    if knowledge_base:
        logger.info("✅ Knowledge base initialized successfully")
        knowledge_base_status = "active"
    else:
        logger.warning("⚠️ Knowledge base initialization failed")
        knowledge_base_status = "limited"
except Exception as e:
    logger.error(f"❌ Knowledge base error: {e}")
    knowledge_base = None
    knowledge_base_status = "error"

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "name": "NeoMed Digital Healthcare API",
        "version": "2.0.0",
        "description": "واجهة برمجية موحدة للروبوت الطبي الذكي",
        "documentation": "/api/docs",
        "gradio_interface": "/gradio",
        "endpoints": {
            "chat": "/api/chat",
            "chat_stream": "/api/chat/stream",
            "upload": "/api/upload",
            "clear_memory": "/api/clear-memory",
            "session_info": "/api/session/{session_id}",
            "status": "/api/status"
        }
    }

# ==========================================
# 1. Chat Endpoint (equivalent to chat_function_wrapper)
# ==========================================

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def api_chat(chat_request: ChatRequest):
    """
    💬 الدردشة مع الروبوت الطبي
    
    يقابل: chat_function_wrapper + chat_function_streaming
    """
    try:
        # Validate input
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(400, "الرسالة فارغة")
        
        # Get or create session
        session_id = chat_request.session_id or generate_session_id()
        session = get_api_session(session_id)
        
        # Prepare message with document context
        message_to_agent = chat_request.message
        if session["processed_docs"]:
            str_processed_docs = "\n".join([
                f"{doc.page_content}\n{doc.metadata}" 
                for doc in session["processed_docs"]
            ])
            message_to_agent = f"{chat_request.message}\n\nThis is Information you can use:\n\n{str_processed_docs}"
        
        # Get response from agent
        logger.info(f"💬 Processing chat for session {session_id[:8]}...")
        
        # Collect streaming response
        full_response = ""
        async for chunk in safe_run_agent_streaming(message_to_agent):
            full_response += chunk
        
        # Save to session history
        session["chat_history"].append({
            "user": chat_request.message,
            "assistant": full_response,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            response=full_response,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(500, f"خطأ في معالجة الرسالة: {str(e)}")

# ==========================================
# 2. Chat Stream Endpoint (streaming version)
# ==========================================

@app.post("/api/chat/stream", tags=["Chat"])
async def api_chat_stream(chat_request: ChatRequest):
    """
    📡 الدردشة مع البث المباشر
    
    يقابل: chat_function_streaming with SSE
    """
    try:
        if not chat_request.message or not chat_request.message.strip():
            raise HTTPException(400, "الرسالة فارغة")
        
        session_id = chat_request.session_id or generate_session_id()
        session = get_api_session(session_id)
        
        # Prepare message
        message_to_agent = chat_request.message
        if session["processed_docs"]:
            str_processed_docs = "\n".join([
                f"{doc.page_content}\n{doc.metadata}" 
                for doc in session["processed_docs"]
            ])
            message_to_agent = f"{chat_request.message}\n\n{str_processed_docs}"
        
        async def event_generator():
            """SSE generator"""
            full_response = ""
            
            try:
                logger.info(f"📡 Streaming for session {session_id[:8]}...")
                
                async for chunk in safe_run_agent_streaming(message_to_agent):
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id}, ensure_ascii=False)}\n\n"
                
                # Save to history
                session["chat_history"].append({
                    "user": chat_request.message,
                    "assistant": full_response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send completion
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"❌ Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
        raise HTTPException(500, str(e))
# ==========================================
# 3. upload file Endpoint
# ==========================================
    
@app.post("/api/upload", response_model=FileUploadResponse, tags=["Documents"])
async def api_upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    📁 رفع وثيقة طبية
    
    يقابل: upload_and_process_file
    """
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                400,
                f"نوع الملف غير مدعوم: {file_ext}. الأنواع المدعومة: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (10MB limit)
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(400, "الملف كبير جداً. الحد الأقصى 10 ميجابايت")
        
        # ⭐ FIX: Use cross-platform temp directory
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "medical_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        # Process file
        logger.info(f"📁 Processing file: {file.filename}")
        new_documents = process_uploaded_file(temp_file_path)
        
        if not new_documents:
            temp_file_path.unlink()
            raise HTTPException(400, f"لم يتم العثور على محتوى في '{file.filename}'")
        
        # Save to session
        session_id = session_id or generate_session_id()
        session = get_api_session(session_id)
        session["processed_docs"].extend(new_documents)
        
        # Delete temp file
        temp_file_path.unlink()
        
        logger.info(f"✅ File processed: {file.filename} ({len(new_documents)} docs)")
        
        return FileUploadResponse(
            success=True,
            message=f"تم معالجة '{file.filename}' بنجاح",
            session_id=session_id,
            documents_count=len(new_documents),
            filename=file.filename,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(500, f"خطأ في رفع الملف: {str(e)}")
# ==========================================
# 4. Clear Memory Endpoint (equivalent to clear_chat_memory_and_history)
# ==========================================

@app.post("/api/clear-memory", response_model=ClearMemoryResponse, tags=["Session"])
async def api_clear_memory(session_id: Optional[str] = None):
    """
    🗑️ مسح الذاكرة وبدء محادثة جديدة
    
    يقابل: clear_chat_memory_and_history
    """
    try:
        # Clear global memory (for Gradio compatibility)
        clear_memory()
        
        # Clear specific session if provided
        if session_id and session_id in api_sessions:
            api_sessions[session_id]["chat_history"] = []
            api_sessions[session_id]["processed_docs"] = []
            message = f"تم مسح ذاكرة الجلسة {session_id[:8]}... بنجاح"
        else:
            # Clear global processed_docs for Gradio
            global processed_docs
            processed_docs = []
            message = "تم مسح الذاكرة العامة بنجاح. بدأت محادثة جديدة!"
        
        logger.info(f"🗑️ Memory cleared: {session_id or 'global'}")
        
        return ClearMemoryResponse(
            success=True,
            message=message,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Clear memory error: {e}")
        raise HTTPException(500, f"خطأ في مسح الذاكرة: {str(e)}")

# ==========================================
# 5. Session Info Endpoint
# ==========================================

@app.get("/api/session/{session_id}", tags=["Session"])
async def api_get_session_info(session_id: str):
    """
    ℹ️ معلومات الجلسة
    
    endpoint جديد لعرض معلومات الجلسة
    """
    try:
        if session_id not in api_sessions:
            raise HTTPException(404, "الجلسة غير موجودة")
        
        session = api_sessions[session_id]
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "messages_count": len(session["chat_history"]),
            "documents_count": len(session["processed_docs"]),
            "chat_history": session["chat_history"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# ==========================================
# 6. System Status Endpoint (equivalent to validate_startup)
# ==========================================

@app.get("/api/status", response_model=SystemStatusResponse, tags=["System"])
async def api_system_status():
    """
    📊 حالة النظام
    
    يقابل: validate_startup + معلومات إضافية
    """
    try:
        # Check environment variables
        required_vars = ["OPENAI_API_KEY"]
        env_status = all(os.getenv(var) for var in required_vars)
        
        # Count active sessions
        active_count = sum(
            1 for s in api_sessions.values()
            if (datetime.now() - s["last_activity"]).seconds < 3600
        )
        
        return SystemStatusResponse(
            status="operational" if env_status else "degraded",
            knowledge_base_status=knowledge_base_status,
            active_sessions=active_count,
            total_sessions=len(api_sessions),
            uptime="متاح",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        raise HTTPException(500, str(e))

# ==========================================
# 7. Health Check Endpoint
# ==========================================

@app.get("/api/health", tags=["System"])
async def api_health_check():
    """
    💚 فحص صحة النظام
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ==========================================
# FastAPI Startup & Shutdown
# ==========================================

@app.on_event("startup")
async def startup_event():
    """تهيئة عند بدء التشغيل"""
    logger.info("=" * 60)
    logger.info("🚀 Starting NeoMed Digital Healthcare")
    logger.info("=" * 60)
    logger.info(f"✅ API ready at http://localhost:7860")
    logger.info(f"📚 Documentation at http://localhost:7860/api/docs")
    logger.info(f"🎨 Gradio UI at http://localhost:7860/gradio")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """تنظيف عند الإغلاق"""
    logger.info("🛑 Shutting down...")

# ==========================================
# Gradio Interface (Legacy/Optional)
# ==========================================

async def chat_function_streaming(message: str, history: list):
    """
    Gradio chat function - الآن يستخدم نفس المنطق مع API
    """
    if not message or not message.strip():
        history.append([message, "عذراً، لم أتلقَ أي سؤال."])
        yield history, ""
        return
    
    history.append([message, ""])
    
    try:
        global processed_docs
        
        # Prepare message
        message_to_agent = message
        if processed_docs:
            str_processed_docs = "\n".join([
                f"{doc.page_content}\n{doc.metadata}" 
                for doc in processed_docs
            ])
            message_to_agent = f"{message}\n\nThis is Information you can use:\n\n{str_processed_docs}"
        
        # Stream response
        accumulated_response = ""
        async for chunk in safe_run_agent_streaming(message_to_agent):
            accumulated_response += chunk
            history[-1][1] = accumulated_response
            yield history, ""
            
    except Exception as e:
        logger.error(f"❌ Gradio chat error: {e}")
        history[-1][1] = f"عذراً، حدث خطأ: {str(e)}"
        yield history, ""

def upload_and_process_file(file) -> str:
    """
    Gradio file upload - الآن يستخدم نفس المنطق مع API
    """
    global processed_docs
    
    if file is None:
        return "لم يتم رفع أي ملف"
    
    try:
        file_path = Path(file)
        
        # Validate
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc',".jpg", ".jpeg", ".png", ".webp", 
                        ".bmp", ".tiff", ".gif"}
        if file_path.suffix.lower() not in allowed_extensions:
            return f"نوع الملف غير مدعوم: {file_path.suffix}"
        
        # Check size
        file_size = file_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            return "الملف كبير جداً. الحد الأقصى 10 ميجابايت."
        
        # Process
        new_documents = process_uploaded_file(file_path)
        
        if new_documents:
            processed_docs.extend(new_documents)
            return f"✅ تم معالجة '{file_path.name}' بنجاح. أضيفت {len(new_documents)} وثيقة."
        else:
            return f"⚠️ لم يتم العثور على محتوى في '{file_path.name}'"
        
    except Exception as e:
        logger.error(f"❌ File upload error: {e}")
        return f"❌ خطأ: {str(e)}"

def clear_chat_memory_and_history():
    """
    Gradio clear memory - الآن يستخدم نفس المنطق مع API
    """
    global processed_docs
    
    try:
        clear_memory()
        processed_docs = []
        logger.info("✅ Memory cleared (Gradio)")
        return [], "✅ تم مسح الذاكرة بنجاح. بدأت محادثة جديدة!", ""
    except Exception as e:
        logger.error(f"❌ Clear error: {e}")
        return [], f"❌ خطأ: {str(e)}", ""

def chat_function_wrapper(message, history):
    """Gradio wrapper للـ async streaming"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            async_gen = chat_function_streaming(message, history)
            
            while True:
                try:
                    result = loop.run_until_complete(async_gen.__anext__())
                    yield result
                except StopAsyncIteration:
                    break
                    
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"❌ Wrapper error: {e}")
        try:
            global processed_docs
            message_to_agent = message
            if processed_docs:
                str_processed_docs = "\n".join([
                    f"{doc.page_content}\n{doc.metadata}" 
                    for doc in processed_docs
                ])
                message_to_agent = f"{message}\n\n{str_processed_docs}"
            
            response = asyncio.run(safe_run_agent(message_to_agent))
            history.append([message, response])
            yield history, ""
        except Exception as fallback_error:
            history.append([message, f"❌ Error: {str(fallback_error)}"])
            yield history, ""

def create_interface():
    """Create Gradio interface"""
    
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    
    * { font-family: 'Roboto', sans-serif; }
    .rtl { direction: rtl; text-align: right; }
    textarea { font-size: 18px !important; }
    .message { font-size: 18px !important; line-height: 1.6; }
    """
    
    with gr.Blocks(
        title="نيو ميد الرقمية للرعاية الصحية", 
        css=custom_css,
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown("""
            <div style="text-align: center; direction: rtl;">
            # 🏥 نيو ميد الرقمية للرعاية الصحية
            ### المساعد الطبي الذكي
            </div>
        """, elem_classes="rtl")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💬 المحادثة",
                    height=600,
                    rtl=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="رسالتك",
                        placeholder="اطرح سؤالاً...",
                        scale=4,
                        rtl=True
                    )
                    submit_btn = gr.Button("إرسال", variant="primary", scale=1)
                
                gr.Markdown("""
                    <div style="text-align: center; direction: rtl; color: #666;">
                    <em>مشروع تجريبي تعليمي | في الطوارئ: 997</em>
                    </div>
                """, elem_classes="rtl")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📁 رفع الوثائق", elem_classes="rtl")
                
                file_upload = gr.File(
                    label="ارفع وثيقة أو صورة",
                    file_types=[
                        ".pdf", ".txt", ".docx", ".doc",      # Documents
                        ".jpg", ".jpeg", ".png", ".webp",     # Images ✅
                        ".bmp", ".tiff", ".gif"               # More images ✅
                    ],
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="الحالة", 
                    interactive=False,
                    rtl=True
                )
                
                gr.Markdown("### 🧠 إدارة الذاكرة", elem_classes="rtl")
                clear_btn = gr.Button("🗑️ مسح الذاكرة", variant="secondary")
                clear_status = gr.Textbox(
                    label="الحالة", 
                    interactive=False,
                    rtl=True
                )
                
                gr.Markdown("""
                    <div style="direction: rtl;">
                    **ℹ️ معلومات:**
                    - API متاح على /api/docs
                    - دعم عربي/إنجليزي
                    - متاح 24/7
                    </div>
                """, elem_classes="rtl")
        
        # Event handlers
        msg.submit(
            chat_function_wrapper,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        submit_btn.click(
            chat_function_wrapper,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        file_upload.upload(
            upload_and_process_file,
            inputs=file_upload,
            outputs=upload_status
        )
        
        clear_btn.click(
            clear_chat_memory_and_history,
            inputs=[],
            outputs=[chatbot, clear_status, upload_status]
        )
    
    return interface

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    # Validate startup
    required_vars = ["OPENAI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"❌ Missing: {', '.join(missing)}")
        raise ValueError(f"Missing environment variables: {missing}")
    
    logger.info("✅ Validation passed")
    
    # Create Gradio interface
    gradio_interface = create_interface()
    
    # Mount Gradio on FastAPI
    app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")
    
    # Run server
    logger.info("🚀 Starting server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        log_level="info"
    )
