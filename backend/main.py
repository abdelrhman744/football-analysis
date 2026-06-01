"""
Football Match Analysis - FastAPI Backend
Main entry point for the application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routes import video, analysis, chatbot, match_stats

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

app = FastAPI(
    title="Football Match Analysis API",
    description="Backend API for AI-powered football match analysis",
    version="1.0.0",
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded videos and results
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Register API routes
app.include_router(video.router, prefix="/api/video", tags=["Video"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(chatbot.router, prefix="/api/chat", tags=["Chatbot"])
app.include_router(match_stats.router, prefix="/api/match-stats", tags=["Match Stats"])


@app.get("/")
async def root():
    return {
        "message": "Football Match Analysis API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
