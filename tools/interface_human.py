from langchain.tools import Tool, BaseTool
from dotenv import load_dotenv
import os

load_dotenv()

class InteractHuman(BaseTool):
    name = "InteractHuman"
    description = """
    when you need to ask human a question or given him information or any interaction with human
    Input to this tool will be 
    <tool_input>text to send to human</tool_input>
    """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _run(self, tool_input):
        try:
            print("**************human tool")
            print("tool input", tool_input)
            prompt = tool_input

            return prompt
        except Exception as e:
            print(e)


human_tool = Tool(
    name=InteractHuman().name,
    description=InteractHuman().description,
    func=InteractHuman().run
)
