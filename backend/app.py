from contextlib import asynccontextmanager
from fastapi import FastAPI
from routers import ocr, rag, download
from pgdb.pgdb_manager import pg_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the database connection pool
    try:
        pg_manager.connect()
    except Exception as e:
        print(f"Failed to connect to DB: {e}")
        raise e
    yield
    # Close the database connection pool
    pg_manager.disconnect()

app = FastAPI(lifespan=lifespan)

app.include_router(ocr.router)
app.include_router(rag.router)
app.include_router(download.router)

@app.get("/")
def read_root():
    return {"message": "PaddleOCR API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=14321)
