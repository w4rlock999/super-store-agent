from flask import Flask, request, jsonify
import dotenv
from crewai import Agent, Crew, Process, Task, LLM 
from crewai.memory.external.external_memory import ExternalMemory
import dotenv
import os
import mlflow

from custom_storage import CustomStorage

app = Flask(__name__)
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat_history_mem = CustomStorage()

mlflow.crewai.autolog()

# # Optional: Set a tracking URI and an experiment name if you have a tracking server
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("CrewAI-basic-chatbot")

@app.route('/agent-invoke', methods=['POST'])
def agent_invoke():

    try:

        data = request.get_json()
        user_input = data.get('text', '')

        print(f"got message: {user_input}")
        
        chat_history_mem.save("user", user_input)

        llm_openai = LLM(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.2,
            )

        interface_agent = Agent(
            role='Interface Agent',
            goal=f'You are an interface agent that can answer questions and help with tasks. Always try write output in a nice markdown format.',
            backstory="""You are an interface agent that can answer questions and help with tasks. Always try write output in a nice markdown format.""",
            allow_delegation=False,
            llm=llm_openai,
        )

        history = chat_history_mem.search("")
        print(f"History: {history}")

        task1 = Task(
            description=
                """
                You are an interface agent that can answer questions and help with tasks. Always try write output in a nice markdown format. 
                This is the current user query: {user_input}

                This is the chat history: {history}
                """,
            agent=interface_agent,
            expected_output="A nice markdown format response to the user query, just the markdown without any triple quotes or markdown tags",
        )

        crew = Crew(
            agents=[interface_agent],
            tasks=[task1],
            process=Process.sequential,
        )

        result = crew.kickoff({"user_input": user_input, "history": history})
        print(result.json_dict)

        chat_history_mem.save("assistant", result.raw)

        AI_response = result
        return jsonify({"response": AI_response.raw}), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)