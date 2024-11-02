
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from agent import chain_with_history_and_agent

# Constants
APP_TITLE = "LangChain Server"
APP_VERSION = "1.0"
APP_DESCRIPTION = "Spin up a simple api server using LangChain's Runnable interfaces"
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
)


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

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)
