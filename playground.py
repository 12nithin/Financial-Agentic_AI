from phi.agent import Agent
import phi.api
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

## Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

## Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True, 
            analyst_recommendations=True, 
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Create playground
playground = Playground(agents=[finance_agent, web_search_agent])
app = playground.get_app()

if __name__ == "__main__":
    # Alternative 1: Direct app run
    serve_playground_app(app, reload=True)
    
    # Alternative 2: If the above doesn't work, use this:
    # import uvicorn
    # uvicorn.run(app, host="localhost", port=7777, reload=True)