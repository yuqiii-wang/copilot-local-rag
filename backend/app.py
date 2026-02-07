from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import ocr, rag, download, data_router
from data_manager import data_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize offline manager (creates dirs if needed)
    try:
        data_manager.connect()
    except Exception as e:
        print(f"Error initializing data_manager: {e}")

    # Load Knowledge Graph Model (Async)
    try:
        from services.knowledge_graph_service import kg_service
        await kg_service.load_async()
    except Exception as e:
        print(f"Error loading Knowledge Graph model: {e}")
        
    yield
    # No real disconnect needed
    data_manager.disconnect()

app = FastAPI(lifespan=lifespan)

app.include_router(ocr.router)
app.include_router(rag.router)
app.include_router(download.router)
app.include_router(data_router.router, prefix="/data", tags=["data"])

# Admin UI
from routers import admin as admin_router
app.include_router(admin_router.router, prefix="/admin", tags=["admin"])

# Serve admin static files (graph PNG) by mounting the folder
from fastapi.staticfiles import StaticFiles
app.mount('/admin/static', StaticFiles(directory='admin'), name='admin-static')

@app.get("/")
def read_root():
    return {"message": "PaddleOCR API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=14321)
