from flask import Flask, request, jsonify
from typing import Dict, List, Tuple, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph, END
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from research_agent import ResearchAgent
from langchain_core.tools import tool
# from basic_tool_node import BasicToolNode
from langchain_tavily import TavilySearch


import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)


# Define tools as standalone functions
@tool
def get_top_product_in_month_year(month_year: str) -> str:
    """Get the top performing product at a given month, month must be formatted as YYYY-MM"""

    print(month_year)

    return "Black Hoodie"

class InternalThoughtAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self._create_agent()
        
    class AgentState(MessagesState):
        scratchpad : str    
    
    def main_node(self, state: AgentState):

        basic_prompt_string = f"""
            You are an internal thought agent that is to help the main agent to think and take note about their task to build travel itinerary.
            Always take note and update of the current user data and also your relevant thought to satisfy our specific high standard of data gathering.
            
            Information to gather to build the final travel itinerary:
            1. location, need to be specific, if user mention multiple locations, need to think how to divide them into distinct location to focus the itinerary to e.g. list of cities, list of island
            2. when, need to be in the form of number of days, we need granularity, insist on asking number of days, or translate user's answer to a suggestion of number of days but you need user confirmation of aproval of your suggestion, unless user explicitly say they don't know then you can just approximate it but make sure to say it to user
            3. user preference for the trip, need to know each of this categories: type of leisure, type of activity, type of location, type of food, and their degree of preference to each,  before moving on, do not skip unless user explicitly say that they still need time, user might just sprouting anything at first, but keep in mind that they might change their mind later on, main agent need to help user to figure out what they really love to
            4. list of tourism attractions that user want, this is the meat of the conversation, give user recommendations AND reference, keep in mind user preferences, discuss further with references knowledge
            5. accomodation location, this info should in accordance of the list of the locations and activites, it might be best to stay in one location, it might be best to change accomodation, asks the user what his preferences
            6. transportation modes, should be in accordance of the list of the locations and activites, and the accomodation for each days
            
            You need to constantly think the feasibility of the plan and write it in the scratchpad to give thought to the main agent.

            This is the current scratchpad:
            {state["scratchpad"]}

            These are the latest messages from the user and main agent, you need to output an updated scratchpad, take note of the latest message, write your thoughts, the updated scratchpad will be given to the main agent to drive the next step:
            {state["messages"]}
            """
        
        system_prompt = SystemMessage(content=basic_prompt_string)    
        messages = [system_prompt]

        # print(f"internal thought agent messages: {messages}")
        response = self.llm.invoke(messages)

        return {"scratchpad": response.content}

    def _create_agent(self):
        graph = StateGraph(self.AgentState)
        
        # Add the model node
        graph.add_node("model", self.main_node)
        graph.set_entry_point("model")
        
        # Always END after model
        graph.add_edge("model", END)
        
        # self.checkpointer = MemorySaver()
        # graph = graph.compile(checkpointer=self.checkpointer)
        agent_graph = graph.compile()
        
        return agent_graph
    

class MainAgent:

    def __init__(self, llm, internal_thought_agent):
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))


        self.tavily_search_tool = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5
        )

        # self.tools = [self.tavily_search_tool, get_top_product_in_month_year]
        self.tools = [get_top_product_in_month_year]

        # llm_model = llm

        self.llm_with_tools = llm.bind_tools(self.tools)
        self.llm = llm.bind_tools(self.tools)

        # self.tool_node = BasicToolNode(tools=self.tools)
        self.tool_node = ToolNode([get_top_product_in_month_year])

        self.internal_thought_agent = internal_thought_agent
        self.graph = self._create_agent()

    class AgentState(MessagesState):
        scratchpad : str

    basic_prompt_string = f"""
    You are a helpful assistant that can answer questions and help with tasks. 
    Always try write output in a nice markdown format. 
    """
    
    agent_prompt_string = instructions = f"""
    Your name is "Shopify Admin Agent", an ai agent that is a helpful shopify admin assistant with the ability to search the web and use other tools such as querying admin data from shopify.

    You have access to the following tools:
    - get_top_product_in_month_year: Get the top performing product at a given month (format: YYYY-MM)

    IMPORTANT: When a user asks about product performance, sales data, or anything related to business metrics, you MUST use the available tools to get accurate information. Do not make up data or estimates. Always use the tools when appropriate.

    Examples of when to use tools:
    - User asks: "What was our top product in January 2024?" → Use get_top_product_in_month_year with "2024-01"
    - User asks: "Show me the best selling item last month" → Use get_top_product_in_month_year with the appropriate month
    - User asks: "Which product performed best in Q1?" → Use get_top_product_in_month_year for each month in Q1

    When you use a tool, explain what you're doing to the user and then provide the results in a clear, helpful format.
    """

    system_prompt = SystemMessage(content=basic_prompt_string + agent_prompt_string)    

    def update_scratchpad(self, state: AgentState):
        messages = state["messages"]
        if len(messages) >= 2:
            response = self.internal_thought_agent.graph.invoke({"messages": [messages[-2], messages[-1]], "scratchpad": state.get("scratchpad", "")})
        else:
            response = self.internal_thought_agent.graph.invoke({"messages": [messages[-1]], "scratchpad": state.get("scratchpad", "")})     

        # print(f"internal thought agent response: {response}")
        print(f"internal thought agent updated scratchpad: {response['scratchpad']}")
        return response["scratchpad"]

    def main_node(self, state: AgentState):
        # updated_scratchpad = self.update_scratchpad(state)
        # messages = [self.system_prompt] + [updated_scratchpad] + state["messages"]
        # response = self.llm.invoke(messages)

        # return {"messages": [AIMessage(content=response.content)], "scratchpad":updated_scratchpad}

        messages = [self.system_prompt] + state["messages"]
        response = self.llm.invoke(messages)

        # return {"messages": [AIMessage(content=response.content)], "scratchpad": ""}
        return {"messages": [response], "scratchpad": ""}

    def path_tool_model(self, state: MessagesState):
        
        # tool_route = BasicToolNode.route_tools(state)

        # if tool_route :
        #     print("call tool....")
        #     return "tools"
        
        # return END

        messages = state["messages"]
        last_message = messages[-1]
        print("last message: ", last_message)
        if last_message.tool_calls:
            print("calling tool...")
            return "tools"
        return END

    def _create_agent(self):
        graph = StateGraph(self.AgentState)
        
        # Add the model node
        graph.add_node("model", self.main_node)
        graph.add_node("tools", self.tool_node)
        graph.set_entry_point("model")
        
        graph.add_conditional_edges("model", self.path_tool_model, ["tools", END])
        graph.add_edge("tools", "model")
        
        self.checkpointer = MemorySaver()
        agent_graph = graph.compile(checkpointer=self.checkpointer)
        
        return agent_graph

# Initialize the agents
internal_thought_agent = InternalThoughtAgent(llm)
main_agent = MainAgent(llm, internal_thought_agent)
research_agent = ResearchAgent(llm)

@app.route('/agent-invoke', methods=['POST'])
def agent_invoke():
    try:
        # Parse incoming JSON
        data = request.get_json()
        text = data.get('text', '')

        print(f"got message: {text}")

        agent_config = {
            "configurable": {
                "thread_id": "1"
            }
        }

        # Process the text through the agent
        try:
            
            response = main_agent.graph.invoke({"messages": [HumanMessage(content=text)]}, config=agent_config)
            AI_response = response["messages"][-1].content

            return jsonify({"response": AI_response}), 200
        
        except Exception as invoke_error:
            print(f"Error during agent invocation: {str(invoke_error)}")
            return jsonify({"error": f"Agent error: {str(invoke_error)}"}), 500

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

