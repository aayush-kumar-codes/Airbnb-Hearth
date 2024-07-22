from Agent import AirbnbAgent
from tools.knowledgebase_tool import knowledge_tool
from llm_model import chat_model

agent = AirbnbAgent(llm=chat_model,tools=[knowledge_tool])


agent.run("i am looking two sized beds in Los angeles state")

