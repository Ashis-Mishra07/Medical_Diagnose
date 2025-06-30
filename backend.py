from fastapi import FastAPI, HTTPException, File, Response
from fastapi.responses import FileResponse
import os
from pathlib import Path
from dotenv import load_dotenv

from models import NewsRequest
from utils import generate_broadcast_news, tts_to_audio  # Removed text_to_audio_elevenlabs_sdk
from news_scraper import NewsScraper
from reddit_scraper import scrape_reddit_topics

app = FastAPI()
load_dotenv()

# Add these endpoints to your backend.py

@app.get("/")
async def root():
    """Root endpoint for basic health check"""
    return {"message": "Medical News API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "Medical News Backend",
        "endpoints": ["/generate-news-audio", "/health"],
        "message": "Backend is ready to process medical news requests"
    }


@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        print(f"Processing request for topics: {request.topics}")
        results = {}
        
        if request.source_type in ["news", "both"]:
            print("Scraping news...")
            try:
                news_scraper = NewsScraper()
                results["news"] = await news_scraper.scrape_news(request.topics)
                print(f"News results: {results['news']}")
            except Exception as e:
                print(f"News scraping failed: {str(e)}")
                results["news"] = {"news_analysis": {topic: f"News unavailable: {str(e)}" for topic in request.topics}}
        
        if request.source_type in ["reddit", "both"]:
            print("Scraping reddit...")
            try:
                results["reddit"] = await scrape_reddit_topics(request.topics)
                print(f"Reddit results: {results['reddit']}")
            except Exception as e:
                print(f"Reddit scraping failed: {str(e)}")
                results["reddit"] = {"reddit_analysis": {topic: f"Reddit unavailable: {str(e)}" for topic in request.topics}}

        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})

        # Fallback if both sources fail
        if not news_data and not reddit_data:
            fallback_text = f"I apologize, but I'm currently unable to fetch the latest information about {', '.join(request.topics)}. Please try again later."
        else:
            print("Generating broadcast news...")
            fallback_text = generate_broadcast_news(
                api_key=os.getenv("GROQ_API_KEY2"),
                news_data=news_data,
                reddit_data=reddit_data,
                topics=request.topics
            )
        
        print(f"Generated summary: {fallback_text[:100]}...")

        print("Converting to audio...")
        audio_path = tts_to_audio(
            text=fallback_text,
            language="en"
        )
        print(f"Audio saved to: {audio_path}")

        if audio_path and Path(audio_path).exists():
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=news-summary.mp3"}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
    
    except Exception as e:
        print(f"Error in generate_news_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=1234,
        reload=True
    )