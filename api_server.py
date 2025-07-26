# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from fastapi.responses import JSONResponse
# from main_agent import run_agent
# from typing import Optional

# app = FastAPI()

# # Optional: enable CORS for your frontend to access it
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     user_id: str
#     message: str
#     summary: Optional[str] = None

# @app.post("/chat")
# async def chat_endpoint(data: ChatRequest):
#     # Your LangChain logic here
#     user_id = data.user_id
#     message = data.message
#     summary = data.summary or ""

#     # Example dummy response (replace with LangChain response)
#     langchain_response = run_agent(message)

#     return JSONResponse(content={"response": langchain_response})
