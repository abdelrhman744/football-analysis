"""
Chatbot routes.
"""

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services import chatbot_service

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a question about a football match and receive an AI-powered response.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        response = await chatbot_service.generate_chat_response(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

    return response
