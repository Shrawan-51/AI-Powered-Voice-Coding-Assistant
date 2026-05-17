from langgraph.graph import StateGraph, MessagesState, START, END
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode , tools_condition
from langchain.messages import SystemMessage
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    api_key=GEMINI_API_KEY
)


class State(TypedDict):
    messages : Annotated[list,add_messages]

@tool
def run_command(com:str):
    """
    You are on windows system and this function requires windows command to execute.
    Takes a command line prompt and executes it on the user's machine and 
    returns the output of the command.
    Example: run_command(cmd="ls") where ls is the command to list the files.
    """
    # result = os.system(command=com)
    blocked = ["del", "format", "shutdown", "rmdir", "taskkill"]

    if any(b in com.lower() for b in blocked):
        return "Blocked dangerous command."
    try:
        result = subprocess.run(
            com,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10          # prevent hanging commands
        )
        output = result.stdout.strip()
        error  = result.stderr.strip()
 
        if output and error:
            return f"STDOUT:\n{output}\n\nSTDERR:\n{error}"
        return output or error or "(command ran with no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"
    # return result

@tool
def read_file(path: str) -> str:
    """
    Reads and returns the contents of a file at the given path.
    Use this to inspect existing code before modifying it.
    Example: read_file(path="C:/projects/main.py")
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found at '{path}'."
    except Exception as e:
        return f"Error reading file: {e}"
 
 
@tool
def write_file(path: str, content: str) -> str:
    """
    Writes (or overwrites) a file at the given path with the provided content.
    Creates any missing parent directories automatically.
    Example: write_file(path="C:/projects/Hello.java", content="public class Hello {...}")
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written successfully: {path}"
    except Exception as e:
        return f"Error writing file: {e}"
 
tools = [run_command,read_file,write_file]

llm_with_tool = model.bind_tools(tools=tools)


def chatnode(state:State):
    systemprompt = SystemMessage(content="""
        You are an AI Coding assistant who takes an input from user and based on available
        tools you choose the correct tool and execute the commands.
        You can even execute commands and help user with the output of the command.

        Important instructions:
        - write you generated all files inside the LLM_output folder 
        - Prefer single complete commands instead of many small commands.
        - Never create files line-by-line using repeated echo commands.
        - If creating code files, generate the full content in one command.
        - After receiving tool output, respond normally to the user.
        - Do not repeatedly call tools.
        - Only call tools when necessary.                     
        you are running on windows machine so you can only execute windows commands. 
        DO NOT use Linux commands like mkdir -p,cat <<EOF,touch etc                                    
        Follow these guidelines. 
        """)
    messages = state["messages"][-8:]
    try:
        response = llm_with_tool.invoke([systemprompt]+messages)
    except Exception as e:
        return {
            "messages":[
                {
                    "role":"assistant",
                    "content":f"LLM error{e}"
                }
            ]
        }
    assert len(response.tool_calls) <= 1
    return {"messages":[response]}


tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)

graph_builder.add_node("chatnode",chatnode)
graph_builder.add_node("tools",tool_node)

graph_builder.add_edge(START,"chatnode")
graph_builder.add_conditional_edges(
    "chatnode",
    tools_condition
)
graph_builder.add_edge("tools","chatnode")
graph_builder.add_edge("chatnode",END)

#graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)