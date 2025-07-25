# class InternalThoughtAgent:
#     def __init__(self, llm):
#         self.llm = llm
#         self.graph = self._create_agent()
        
#     class AgentState(MessagesState):
#         scratchpad : str    
    
#     def main_node(self, state: AgentState):

#         basic_prompt_string = f"""
#             You are an internal thought agent that is to help the main agent to think and take note about their task to build travel itinerary.
#             Always take note and update of the current user data and also your relevant thought to satisfy our specific high standard of data gathering.
            
#             Information to gather to build the final travel itinerary:
#             1. location, need to be specific, if user mention multiple locations, need to think how to divide them into distinct location to focus the itinerary to e.g. list of cities, list of island
#             2. when, need to be in the form of number of days, we need granularity, insist on asking number of days, or translate user's answer to a suggestion of number of days but you need user confirmation of aproval of your suggestion, unless user explicitly say they don't know then you can just approximate it but make sure to say it to user
#             3. user preference for the trip, need to know each of this categories: type of leisure, type of activity, type of location, type of food, and their degree of preference to each,  before moving on, do not skip unless user explicitly say that they still need time, user might just sprouting anything at first, but keep in mind that they might change their mind later on, main agent need to help user to figure out what they really love to
#             4. list of tourism attractions that user want, this is the meat of the conversation, give user recommendations AND reference, keep in mind user preferences, discuss further with references knowledge
#             5. accomodation location, this info should in accordance of the list of the locations and activites, it might be best to stay in one location, it might be best to change accomodation, asks the user what his preferences
#             6. transportation modes, should be in accordance of the list of the locations and activites, and the accomodation for each days
            
#             You need to constantly think the feasibility of the plan and write it in the scratchpad to give thought to the main agent.

#             This is the current scratchpad:
#             {state["scratchpad"]}

#             These are the latest messages from the user and main agent, you need to output an updated scratchpad, take note of the latest message, write your thoughts, the updated scratchpad will be given to the main agent to drive the next step:
#             {state["messages"]}
#             """
        
#         system_prompt = SystemMessage(content=basic_prompt_string)    
#         messages = [system_prompt]

#         # print(f"internal thought agent messages: {messages}")
#         response = self.llm.invoke(messages)

#         return {"scratchpad": response.content}

#     def _create_agent(self):
#         graph = StateGraph(self.AgentState)
        
#         # Add the model node
#         graph.add_node("model", self.main_node)
#         graph.set_entry_point("model")
        
#         # Always END after model
#         graph.add_edge("model", END)
        
#         # self.checkpointer = MemorySaver()
#         # graph = graph.compile(checkpointer=self.checkpointer)
#         agent_graph = graph.compile()
        
#         return agent_graph