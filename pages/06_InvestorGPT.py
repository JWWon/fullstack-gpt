from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import streamlit as st
from langchain.tools import BaseTool
from typing import Type
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage
import yfinance as yf

st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT
            
    Welcome to InvestorGPT.
            
    Write down the name of a company and our Agent will do the research for you.
"""
)


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class StockMarketSymbolSearchTool(BaseTool):
    name: Type[str] = "StockMarketSymbolSearch"
    description: Type[
        str
    ] = """
    Use this tool to search the stock market symbol for a company.
    It takes a query as an argument.
    Example: "Apple" -> "AAPL"
    """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = (
        StockMarketSymbolSearchToolArgsSchema
    )

    def _run(self, query: str) -> str:
        ddgs = DuckDuckGoSearchRun()
        return ddgs.run(query)


class CompanyOverviewToolArgsSchema(BaseModel):
    symbol: str = Field(description="The stock market symbol of the company")


class CompanyOverviewTool(BaseTool):
    name: Type[str] = "CompanyOverview"
    description: Type[str] = "Use this tool to get an overview of a company."
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol: str) -> str:
        company = yf.Ticker(symbol)
        return company.info


class CompanyIncomeStatementToolArgsSchema(BaseModel):
    symbol: str = Field(description="The stock market symbol of the company")


class CompanyIncomeStatementTool(BaseTool):
    name: Type[str] = "CompanyIncomeStatement"
    description: Type[str] = "Use this tool to get an income statement of a company."
    args_schema: Type[CompanyIncomeStatementToolArgsSchema] = (
        CompanyIncomeStatementToolArgsSchema
    )

    def _run(self, symbol: str) -> str:
        company = yf.Ticker(symbol)
        return company.income_stmt


class CompanyStockWeeklyHistoryToolArgsSchema(BaseModel):
    symbol: str = Field(description="The stock market symbol of the company")


class CompanyStockWeeklyHistoryTool(BaseTool):
    name: Type[str] = "CompanyStockWeeklyHistory"
    description: Type[str] = (
        "Use this tool to get the weekly stock history of a company."
    )
    args_schema: Type[CompanyStockWeeklyHistoryToolArgsSchema] = (
        CompanyStockWeeklyHistoryToolArgsSchema
    )

    def _run(self, symbol: str) -> str:
        company = yf.Ticker(symbol)
        return company.history(period="5d")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockWeeklyHistoryTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(
            content="""
            You are a hedge fund manager.
            
            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.
            
            Consider the weekly stock history, the company overview and the income statement.
            
            Be assertive in your judgement and recommend the stock or advise the user against it.
            """
        )
    },
)


company = st.text_input("Enter a company name you want to analyze")
if company:
    result = agent.invoke({"input": company})

    st.write(result["output"].replace("$", "\$"))
