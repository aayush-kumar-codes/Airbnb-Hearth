from langchain.tools import Tool, BaseTool
from langchain_community.embeddings import BedrockEmbeddings
import boto3
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_aws import ChatBedrock
import os

load_dotenv()
region = 'us-east-1'
bedrock_client = boto3.client("bedrock-runtime", aws_access_key_id=os.getenv("aws_access_key"),
                              aws_secret_access_key=os.getenv("aws_secret_key"),
                              region_name=region)

embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

vectorstore_faiss = FAISS.load_local("./data", embedding, allow_dangerous_deserialization=True)


class KnowledgeBase(BaseTool):
    name = "KnowledgeBaseTool"
    description = """
    useful when you are requested to provide the property recommendation based on the user input 

    Input to this tool will be always "search query ","city name" expected input for city name is two letters

    search query should involves neighbourhood, price of properity if any, amenities if any, ratings if any don't miss any of these if they existing in the conversation

    Example: 
    looking for "two rooms hotel with 1 kingsize bed and 1 queensize bed" , LA(city name)

    <tool_input>generated search query in natural language based on conversation,cityname in two letters</tool_input>
    """

    def format_docs(docs):

        return "\n\n".join(doc.page_content for doc in docs)

    def _run(self, tool_input):
        llm = ChatBedrock(
            credentials_profile_name="kamal",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0},
            streaming=True,
            region_name="us-east-1"
        )
        try:
            print("**************knowlegebase tool")
            print("tool input", tool_input, tool_input.split(','))
            splits = tool_input.split(',')
            state = splits[-1]
            query = " , ".join(splits[0:len(splits) - 1])
            print(f"Parsed input - Prompt: '{query}', State: '{state}'")

            filter = {
                "bool": {
                    "must": [
                        {"term": {"metadata.state.keyword": state}}
                    ]
                }
            }

            query_embedding = embedding.embed_query(query)
            relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding, k=10)

            print(f"relevant_documents {len(relevant_documents)}")
            if (len(relevant_documents)) > 0:

                docs = []

                for i, rel_doc in enumerate(relevant_documents):
                    print(f'## Document {i+1}: {rel_doc.metadata} {rel_doc.page_content}')
                    docs.append(f"""<property>{rel_doc.page_content}<listing_id>{rel_doc.metadata["id"]}</listing_id></property>""")

                    # First check the properties against the search query and then filter out properties which don't match the search query.
#
                prompt = f"""
                    You are an Airbnb Agent who recommend the property based on people preferences.
                    User has search for properties using query {query}

                    These are the responses which we have got from database
                    <properties>
                    {"\n".join(docs)}
                    </properties>


                    Generate a well formatted message to display to the users with all information beautifully formatted.
                    Make sure to include all details. Rent and Rating should be highlighted clearly and use tags "Rent" and "Rating"
                    Use html formatting, but make sure not use lot of new line <div> or <br>
                    Also make sure to a hyperlink for the listing in format https://www.airbnb.co.in/rooms/<listing_id> and display the listing id as well
                    Make sure to display listing id and don't not make up listing id. Use the same listing id as provided in the properties.

                    Only show the message for user nothing else.

                    Being:
                    """

                return llm.invoke(prompt).content
            else:
                prompt = f"""
                    You are an Airbnb Agent who recommend the property based on people preferences.
                    User has search for properties using query {query}

                    We were not able to find any properties matching users query.

                    Generate a well formatted message to display to the users the properties based on his preference.
                    Only show the message for user nothing else.
                    Being:
                    """

                return llm.invoke(prompt).content

        except Exception as e:
            print(e)


knowledge_tool = Tool(
    name=KnowledgeBase().name,
    description=KnowledgeBase().description,
    func=KnowledgeBase().run
)
