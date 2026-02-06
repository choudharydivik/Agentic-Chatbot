import os
from tavily import TavilyClient
from langchain_core.prompts import ChatPromptTemplate


class AINewsNode:
    def __init__(self, llm):
        self.tavily = TavilyClient()
        self.llm = llm

    def fetch_news(self, state: dict) -> dict:
        frequency = state["messages"][0].content.lower()
        state["frequency"] = frequency

        time_range_map = {"daily": "d", "weekly": "w", "monthly": "m"}
        days_map = {"daily": 1, "weekly": 7, "monthly": 30}

        response = self.tavily.search(
            query="Top Artificial Intelligence (AI) technology news India and globally",
            topic="news",
            time_range=time_range_map[frequency],
            max_results=20,
            days=days_map[frequency],
        )

        state["news_data"] = response.get("results", [])
        return state

    def summarize_news(self, state: dict) -> dict:
        news_items = state["news_data"]

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """Summarize AI news articles into markdown.
             Format:
             ### [YYYY-MM-DD]
             - [Summary](URL)
             Sort by latest first."""),
            ("user", "Articles:\n{articles}")
        ])

        articles_str = "\n\n".join(
            f"Content: {i.get('content','')}\nURL: {i.get('url','')}\nDate: {i.get('published_date','')}"
            for i in news_items
        )

        response = self.llm.invoke(prompt.format(articles=articles_str))
        state["summary"] = response.content
        return state

    def save_result(self, state: dict) -> dict:
        frequency = state["frequency"]
        summary = state["summary"]

        # 🔥 ONLY IMPORTANT FIX
        os.makedirs("./AINews", exist_ok=True)

        filename = f"./AINews/{frequency}_summary.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# {frequency.capitalize()} AI News Summary\n\n")
            f.write(summary)

        state["filename"] = filename
        return state
