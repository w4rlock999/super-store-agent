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
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from basic_tool_node import BasicToolNode

import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=os.getenv("OPENAI_API_KEY"))

tavily_tool = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=2
        )

class ResearchAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tavily_search_tool = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5
        )
        self.tools = [self.tavily_search_tool]
        self.llm_with_tools = llm.bind_tools(self.tools)

        self.tool_node = BasicToolNode(tools=self.tools)

        self.graph = self._create_agent()

    class AgentState(MessagesState):
        extra_info : str
        task_list : str
        original_task : str
    
    def plan_node(self, state: AgentState):

        original_task = state["messages"][-1].content

        planning_node_prompt = SystemMessage(content=
            f"""
            You are a task breaking down assistant. Your job is to make step or steps, from the current task into smaller tasks and steps for each task.

            For example, if someone asks about interesting places to visit in two different cities, break it down into two steps for each city.
            If they ask about a single location, create just one step.

            Important rules:
            - Always number your steps, even if there's only one
            - Don't execute the tasks yourself
            - Don't ask the user for clarification
            - There should not be redundant steps
            - Try to always make no more than 3 steps per smaller task, unless you really really need to
            

            This is the current task to be deconstructed:
            {state["messages"][-1].content}
            """)
        full_prompt = [planning_node_prompt]

        response = self.llm.invoke(full_prompt)
        print(f"planning node response: {response}")


        return {"messages": [response],
                "task_list": response.content,
                "original_task": original_task}
    
    def action_node(self, state: AgentState):

        action_node_prompt = SystemMessage(content=
            f"""
            You are an assistant that will execute a step. 
            This is the list of steps you must do:
            {state["task_list"]}
            
            Execute the step one by one. Find in the history which action is needed to be done right now. 
            In each generation, except for tool calling, you must begin with the sentence "doing task <step_number>" and continue with doing the current step. <step_number> is the number of the task to be done from the action plan.
            You must end the last step with the sentence "all tasks are done".

            NEVER SAY "all tasks are done" if you are not in the last task.

            This is the execution history:
            """)

        full_prompt = [action_node_prompt] + state["messages"]
        response = self.llm_with_tools.invoke(full_prompt)


        # prompt = PromptTemplate.from_template(action_node_prompt.content)
        # action_agent = create_react_agent(
        #     model=self.llm,
        #     tools=[self.tavily_search_tool],
        #     prompt=prompt
        # )
        # response = action_agent.invoke(state)

        print(f"action node response: {response}")

        return {"messages": [response]}
    
    def finalizer_node(self, state: AgentState):
        finalizer_node_prompt = SystemMessage(content=
            f"""
            You are a finalizer assistant. Your job is to finalize the task by generating a final answer to the original task/question, taking into account of the information you have aquired.
            You must use the information you have aquired to generate a final answer, include all the references links in the final answer!

            This is the current task to be finalized:
            {state["original_task"]}

            This is the execution history:
            """
        )
        full_prompt = [finalizer_node_prompt] + state["messages"]
        response = self.llm.invoke(full_prompt)
        print(f"finalizer node response: {response}")

        return {"messages": [response]}
    
    def to_continue_action_node(self, state: AgentState):

        # route_tools is a method of BasicToolNode class to check if the last message has tool calls
        tool_route = BasicToolNode.route_tools(state)

        if tool_route == False:
            if "all tasks are done" in state["messages"][-1].content.lower():
                return "finalizer"
            else:
                return "action"
        else:
            return "tools"

    def _create_agent(self):
        graph = StateGraph(self.AgentState)
        
        # Add the model node
        graph.add_node("planner", self.plan_node)
        graph.add_node("action", self.action_node)
        graph.add_node("tools", self.tool_node)
        graph.add_node("finalizer", self.finalizer_node)

        graph.set_entry_point("planner")

        graph.add_edge("planner", "action")
        graph.add_conditional_edges("action", self.to_continue_action_node)
        graph.add_edge("tools", "action")
        graph.add_edge("finalizer", END)
        agent_graph = graph.compile()
        return agent_graph
    

