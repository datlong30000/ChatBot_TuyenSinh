import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langserve import add_routes

from agent import chain_with_history_and_agent

# Constants
APP_TITLE = "LangChain Server"
APP_VERSION = "1.0"
APP_DESCRIPTION = "Spin up a simple api server using LangChain's Runnable interfaces"
HOST = "0.0.0.0"
PORT = 8000
MAX_REQUEST_SIZE = 1 * 1024 * 1024  # 1 MB

# Configure rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Add routes for the LangChain agent
add_routes(
    app,
    chain_with_history_and_agent,
)

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    if request.headers.get("content-length") is not None:
        if int(request.headers.get("content-length", 0)) > MAX_REQUEST_SIZE:
            return JSONResponse(status_code=413, content={"error": "Request too large"})
    return await call_next(request)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
@limiter.limit("5/minute")
async def root(request: Request):
    return {"message": "Hello World"}

from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
import asyncio

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
