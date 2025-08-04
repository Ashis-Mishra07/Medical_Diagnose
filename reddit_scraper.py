from typing import List
import os
from utils import *
from typing import List
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from tenacity import (
    retry, 
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta

load_dotenv()

two_weeks_ago = datetime.today() - timedelta(days=14) 
two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')

class MCPOverloadedError(Exception):
    pass

mcp_limiter = AsyncLimiter(1, 15)

# Updated to supported Groq model
model = ChatGroq(
    model="llama3-70b-8192",  # Changed from mixtral-8x7b-32768
    api_key=os.getenv("GROQ_API_KEY2")
)

server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    retry=retry_if_exception_type(MCPOverloadedError),
    reraise=True
)
async def process_topic(agent, topic: str):
    async with mcp_limiter:
        try:
            messages = [{
                "role": "user",
                "content": f"""
                Search for Reddit posts about "{topic}" from the last 2 weeks.
                Focus on:
                1. Recent discussions and trending topics
                2. Public sentiment and opinions
                3. Key insights and developments
                
                Provide a summary of the main discussions and sentiment around {topic}.
                """
            }]
            
            response = await agent.ainvoke({"messages": messages})
            return response["messages"][-1].content
            
        except Exception as e:
            if "overloaded" in str(e).lower() or "rate limit" in str(e).lower():
                raise MCPOverloadedError(f"MCP service overloaded for topic {topic}")
            else:
                return f"Error processing {topic}: {str(e)}"


async def scrape_reddit_topics(topics: List[str]) -> dict[str, dict]:
    """Process list of topics and return analysis results"""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                tools = load_mcp_tools(session)
                agent = create_react_agent(model, tools)
                
                results = {}
                for topic in topics:
                    try:
                        summary = await process_topic(agent, topic)
                        results[topic] = summary
                    except Exception as e:
                        results[topic] = f"Error: {str(e)}"
                
                return {"reddit_analysis": results}
    
    except Exception as e:
        print(f"Reddit scraping failed: {str(e)}")
        # Return empty results instead of failing completely
        return {"reddit_analysis": {topic: f"Reddit unavailable: {str(e)}" for topic in topics}}