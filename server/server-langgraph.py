from flask import Flask, request, jsonify
from typing import Dict, List, Tuple, Any, Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langgraph.graph import START, StateGraph, END, MessagesState
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode, InjectedState
from langgraph.checkpoint.memory import MemorySaver
from research_agent import ResearchAgent
from langgraph.types import Command
from langchain_tavily import TavilySearch
import shopify
import json
import subprocess

import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# ////////////////////////////////////////////////////////////////
# //////////////     Define tool for agents      /////////////////
# ////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

tavily_search_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

@tool 
def get_order_data_for_period(start_date: str, end_date: str, save_to_filename: str, ) -> str:
    """
    Get order data from Shopify Admin API, in a period of time, and save to file name in JSON format.
    start_date and end_date are string in the format of YYYY-MM-DD.
    save_to_filename should be a string with format [start_date]_to_[end_date]_order.json
    save file will be saved in the './runtime_data' directory.

    """

    from datetime import datetime


    def to_iso8601(date_str, is_start=True):
        # Convert from YYYY-MM-DD (or fallback to YYYY-MM) to ISO8601 format
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                dt = datetime.strptime(date_str, "%Y-%m")
                dt = dt.replace(day=1)
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD or YYYY-MM format")
        if is_start:
            return dt.strftime("%Y-%m-%dT00:00:00Z")
        else:
            return dt.strftime("%Y-%m-%dT23:59:59Z")

    start_iso = to_iso8601(start_date, is_start=True)
    end_iso = to_iso8601(end_date, is_start=False)


    store_api_key = os.getenv("SHOPIFY_API_KEY")
    store_secret_key = os.getenv("SHOPIFY_SECRET_KEY")
    admin_api_access_token = os.getenv("SHOPIFY_ADMIN_API_ACCESS_TOKEN")

    shop_url = "urban-thread-test-store.myshopify.com"
    api_version = '2025-07'

    shopify.Session.setup(api_key=store_api_key, secret=store_secret_key)
    session = shopify.Session(shop_url, api_version, admin_api_access_token)
    shopify.ShopifyResource.activate_session(session)

    orders = []
    has_next_page = True
    end_cursor = None

    while has_next_page:
    
        after_clause = f', after: "{end_cursor}"' if end_cursor else ""
        
        query = f"""
        {{
        orders(
            first: 100
            query: "processed_at:>={start_iso} processed_at:<={end_iso}"
            reverse: false
            {after_clause}
        ) {{
            pageInfo {{
            hasNextPage
            endCursor
            }}
            edges {{
            node {{
                id
                name
                processedAt
                totalPriceSet {{
                shopMoney {{
                    amount
                    currencyCode
                }}
                }}
                customer {{
                firstName
                lastName
                email
                }}
                lineItems(first: 5) {{
                edges {{
                    node {{
                    title
                    quantity
                    variant {{
                        id
                        title
                    }}
                    }}
                }}
                }}
            }}
            }}
        }}
        }}
        """

        response = shopify.GraphQL().execute(query)

        data = json.loads(response)

        orders_edges = data["data"]["orders"]["edges"]
        orders.extend(orders_edges)
        page_info = data["data"]["orders"]["pageInfo"]
        has_next_page = page_info["hasNextPage"]
        end_cursor = page_info["endCursor"]

    # Print all orders in January 2024
    filename_to_save = save_to_filename if save_to_filename.endswith('.json') else f"{save_to_filename}.json"
    with open(f"./runtime_data/{filename_to_save}", 'w') as f:
        json.dump(orders, f, indent=2)

    return f"Orders data saved to ./runtime_data/{filename_to_save} for order data from {start_date} to {end_date}"

@tool(parse_docstring=True)
def run_python_code(code: str):
    """
    Run python code, the code is a string of python code, and it will be executed in the current working directory.
    Always print the relevant information with explicity print() statement for any of the result in interest or any error for you to revise the code.
    
    Args:
        code: the python code, always use explicit print() statement to output the desired variable and also the error that may arise. NEVER print a whole file.
        each call is a separate call without any context from the previous run, you should supply the complete code each time you run, especially on import and load data for each run.
        all data MUST be loaded and saved in the './runtime_data' directory and NOT in the current working directory!!!
        
    """
    working_dir = "."

    result = subprocess.run(
        ["python3", "-c", code],
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=5  # optional for safety
    )

    return result.stdout or result.stderr


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:

        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )

        # Append the ToolMessage to the existing messages
        updated_messages = state["messages"] + [tool_message]

        command = Command(
            goto=agent_name,  
            update={**state, "messages": updated_messages}
            # graph=Command.PARENT,  
        )
        
        return command

    return handoff_tool

def pretty_print_message(message, indent=False, agent_name=""):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        print("from agent: ", agent_name)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

class RevenueAnalystAgent:

    def __init__(self, llm):
        self.tools = [run_python_code, tavily_search_tool]

        self.llm_with_tools = llm.bind_tools(self.tools)
        self.llm = llm.bind_tools(self.tools)

        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_agent()

    # TODO actually the manager message will be in teh messages state automatically,
    # let's confirm this
    class AgentState(MessagesState):
        pass

    system_prompt_string = """
        You are revenue analysis agent, you will need to analyse revenue related insight out of sales data.
        The final report agent will call you and give you the relevenat data (its filename, which you can use in the analysis python code)
        
        You need to provide the following report from the data (if it is annual, adapt for quarterly):
        1. total annual revenue
        2. total monthly revenue
        3. quarterly revenue
        4. monthly and quarterly revenue trend

        you can use the tool run_python_code to run code to yield all the above information.

        the data provided to you is a json file from shopify GraphQL, with key "node" and sub keys under it.
        the date for each order is indicated by key "processedAt" NOT "createdAt" as you might understand previously.
        if you see error when running the python code indicating that the structure of the data is different than your thought, try to run a code to understand the structure first.
        
        Do the task ONE BY ONE, generate your thought first, what you are going to do, and then do the task (e.g. using tool) ONLY AFTER you say clearly what you are going to do.

        Calculate every insight in one program at once if you can to be efficient in your work!

        if you have done all the analysis and have written the final data for the final report agent (your supervisor), end the response with this exact string:
        "ALL REVENUE ANALYSIS TASK IS DONE"

        this is the previous messages history:
    """

    def main_node(self, state: AgentState):

        prompt_template = ChatPromptTemplate([
          ("system", self.system_prompt_string)
        ])

        system_prompt = prompt_template.invoke({})

        messages = [SystemMessage(content=system_prompt.to_string())] + state["messages"]
        response = self.llm.invoke(messages)

        pretty_print_message(response, agent_name="revenue analyst")

        return {"messages": [response]}

    def path_tool_model(self, state: AgentState):

        messages = state.get("messages", [])
        if not messages:
            return END
        last_message = messages[-1]
    
        if last_message.tool_calls:
            return "tools"
        elif "ALL REVENUE ANALYSIS TASK IS DONE" in last_message.content:
            return END
        
        return "model"

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
    

class FinalReportAgent:

    def __init__(self, llm):

        self.assign_to_revenue_analyst_agent = create_handoff_tool(
            agent_name="revenue_analyst_agent",
            description="Assign task to a revenue analyst agent.",
        )

        self.revenue_analyst_graph = RevenueAnalystAgent(llm=llm).graph

        self.tools = [self.assign_to_revenue_analyst_agent, get_order_data_for_period, run_python_code, tavily_search_tool]
        # self.tools = [get_order_data_for_period, run_python_code, tavily_search_tool]

        self.llm = llm.bind_tools(self.tools)

        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_agent()
    
    class AgentState(MessagesState):
        pass

    system_prompt_string = """
        You are a supervisor agent for final executive report writing, you will plan, manage agents, assign them to tasks, and compile their output into draft of relevant informations and write the final report. 
        What you need to write is an annual (or quarterly) sales report of a shopify store Urban Thread, selling apparels and accessories.

        For a final annual executive report, it must have all of the item here:
        - full revenue report: total revenue, monthly revenue, trend, quarterly revenue 
        - order report: overall order details, average order per person, trend of number of order, average purchase value per person, purchase trend in a year
        - product performance: top/bottom product from quantity sold, top/bottom product per month, top/bottom product per quarter, top annual revenue contributor product
        
        let's limit the scope to only revenue report and nothing else for now as the system is being built for the other.

        Do this tasks one task at a time:

        1. First you need to make plan on what to do. Summarize the request from the chat history. Write the period of final executive report requested.
        
        2. Retrieve the data you need using the tool get_order_data_for_period for the valid requested period
        
        3. After that, you will then delegate the revenue analysis to the revenue analyst agent, by providing it with the relevant file name for the requested period, and also the list of tasks it must do.
        You need to review the output, refine it, and return it to the revenue analyst agent if it need to redo the task.

        After the revenue analyst agent finish all their task, it will provide you with the final insight.
        You will then provide the final report in a markdown format without any quotes or anything, ready to be rendered.

        If you have done writing the final report and want to pass it to the main agent, write the report in markdown format, ready to be rendered, and start and end with this format:

        start with:
        THIS IS THE FINAL REPORT DRAFT FROM Final Report Agent, PLEASE PRESENT IT TO THE USER, The user won't see this message directly

        end it with:
        THIS IS THE END OF THE SALES REPORT
        to indicate that the writing is finished  

        this is the chat history:
    """

    def main_node(self, state: AgentState):

        prompt_template = ChatPromptTemplate([
          ("system", self.system_prompt_string)
        ])

        # Get the required fields safely with defaults
        messages = state.get("messages", [])
        
        system_prompt = prompt_template.invoke({})

        messages = [SystemMessage(content=system_prompt.to_string())] + messages

        response = self.llm.invoke(messages)

        pretty_print_message(response, agent_name="final report agent")


        return {"messages": [response]}

    def revenue_analyst_agent_node(self, state: AgentState):

        response = self.revenue_analyst_graph.invoke({"messages":state["messages"]})
        # Extract the last message from the response
        final_response = response["messages"][-1]

        return {"messages": [final_response]}
    

    def path_model_tool_END(self, state: AgentState):

        messages = state.get("messages", [])
        if not messages:
            return END
        last_message = messages[-1]
    

        if last_message.tool_calls:
            return "tools"
        elif "THIS IS THE END OF THE SALES REPORT" in last_message.content:
            return END

        return "model"
    
    def path_tool_END(self, state:AgentState):

        messages = state.get("messages", [])
        if not messages:
            return END
        last_message = messages[-1]
    
        # If the last message is a Command, return END
        print("last tool message in final report agent: ", last_message)
        if "Successfully transferred to" in last_message.content:
            print("last instance is handoff")
            return END

        return "model"

    def _create_agent(self):
        graph = StateGraph(self.AgentState)
        
        # Add the model node
        graph.add_node("model", self.main_node)
        graph.add_node("tools", self.tool_node)
        graph.add_node("revenue_analyst_agent", self.revenue_analyst_agent_node)
        graph.set_entry_point("model")
        
        graph.add_conditional_edges("model", self.path_model_tool_END, ["tools", "model", END])
        graph.add_conditional_edges("tools", self.path_tool_END, ["model", END])
        # graph.add_edge("tools", "model")
        # graph.add_edge("tools", "revenue_analyst_agent")
        graph.add_edge("revenue_analyst_agent", "model")
        
        agent_graph = graph.compile()
        
        return agent_graph  

class MainAgent:

    def __init__(self, llm):

        self.final_report_graph = FinalReportAgent(llm=llm).graph

        self.assign_to_final_report_agent = create_handoff_tool(
            agent_name="final_report_agent",
            description="Assign task to a final report agent.",
        )

        self.tools = [get_order_data_for_period, self.assign_to_final_report_agent, run_python_code, tavily_search_tool]

        self.llm_with_tools = llm.bind_tools(self.tools)
        self.llm = llm.bind_tools(self.tools)

        self.tool_node = ToolNode(self.tools)
        self.graph = self._create_agent()

    class AgentState(MessagesState):
        pass
    
    agent_prompt_string = f"""
    You are a helpful assistant that can answer questions and help with tasks. 
    Always try write output in a nice markdown format. 

    Your name is "Shopify Admin Agent", an ai agent that is a helpful shopify admin assistant with the ability to search the web and use other tools such as querying admin data from shopify.

    IMPORTANT: When a user asks about product performance, sales data, or anything related to business metrics, you MUST use the available tools to get accurate information. Do not make up data or estimates. Always use the tools when appropriate.

    Examples of when to use tools:
    - User asks: "make final report for the year 2024?" → handoff to final report agent
    - User asks: "What was our top product in January 2024?" → Use get_top_product_in_month_year with "2024-01"
    - User asks: "Show me the best selling item last month" → Use get_top_product_in_month_year with the appropriate month
    - User asks: "Which product performed best in Q1?" → Use get_top_product_in_month_year for each month in Q1

    When you use a tool, explain what you're doing to the user and then provide the results in a clear, helpful format.

    When user request for a final report, you will know to delegate the work to the final report agent.
    Check the chat history so far, when you see in the chat history that the final report agent already return you the requested final report and you have not present it to the user, you MUST present it to the user if you haven't!

    Here is the chat history so far:
    """

    system_prompt = SystemMessage(content=agent_prompt_string)    

    def main_node(self, state: AgentState):
        
        messages = [self.system_prompt] + state.get("messages", [])
        response = self.llm.invoke(messages)

        pretty_print_message(response, agent_name="main agent")
        # print("DIRTY PRINT: ", response)

        return {"messages": [response]}

    def final_report_agent_node(self, state: AgentState):
        response = self.final_report_graph.invoke({"messages":state["messages"]})
        # Extract the last message from the response
        final_response = response["messages"][-1]

        return {"messages": [final_response]}

    def path_tool_model(self, state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return END
        last_message = messages[-1]
    
        if last_message.tool_calls:
            return "tools"
        return END

    def path_tool_END(self, state:AgentState):

        messages = state.get("messages", [])
        if not messages:
            return END
        last_message = messages[-1]
    
        # If the last message is a Command, return END
        print("last tool message in main agent: ", last_message)
        if "Successfully transferred to" in last_message.content:
            print("last instance is handoff")
            return END

        return "model"

    def _create_agent(self):
        graph = StateGraph(self.AgentState)
        
        # Add the model node
        # TODO rename "model" to main agent
        graph.add_node("model", self.main_node)
        graph.add_node("tools", self.tool_node)
        graph.add_node("final_report_agent", self.final_report_agent_node)
        graph.set_entry_point("model")
        
        graph.add_conditional_edges("model", self.path_tool_model, ["tools", END])
        graph.add_conditional_edges("tools", self.path_tool_END, ["model", END])
        # graph.add_edge("tools", "model")
        graph.add_edge("final_report_agent", "model")
        
        self.checkpointer = MemorySaver()
        agent_graph = graph.compile(checkpointer=self.checkpointer)
        
        return agent_graph

# Initialize the agents
# internal_thought_agent = InternalThoughtAgent(llm)
# main_agent = MainAgent(llm, internal_thought_agent)
main_agent = MainAgent(llm)
research_agent = ResearchAgent(llm)

@app.route('/agent-invoke', methods=['POST'])
def agent_invoke():
    try:
        # Parse incoming JSON
        data = request.get_json()
        text = data.get('text', '')

        agent_config = {
            "configurable": {
                "thread_id": "1"
            }
        }

        # Process the text through the agent
        try:
            response = main_agent.graph.invoke({"messages": [HumanMessage(content=text)]}, config=agent_config)
            
            print("ALL AGENT EXECUTION FINISH, THIS IS THE FINAL MESSAGES LOGS:")

            for message in response["messages"]:
                pretty_print_message(message)

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

