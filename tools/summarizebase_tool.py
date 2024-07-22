from langchain.tools import Tool, BaseTool
from langchain_community.embeddings import BedrockEmbeddings
import boto3
from dotenv import load_dotenv
import os
from langchain_aws import ChatBedrock
import xml.etree.ElementTree as ET
import pandas as pd

load_dotenv()

region = 'us-east-1'

s3_client = boto3.client('s3')

bedrock_client = boto3.client("bedrock-runtime", aws_access_key_id=os.getenv("aws_access_key"),
                              aws_secret_access_key=os.getenv("aws_secret_key"),
                              region_name=region)
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

df = pd.read_csv("reviews.csv")
df['listing_id'] = df['listing_id'].astype(int)


def extract_reviews(listing_id):

    print(df['listing_id'])
    print(df.dtypes)

    listing_id = int(listing_id)

    filtered_df = df[df['listing_id'] == listing_id]

    reviews = filtered_df[['reviewer_name', 'comments']]

    return reviews


def reviews_to_list(reviews_df):
    reviews_list = reviews_df.to_dict(orient='records')
    formatted_reviews = [{'name': review['reviewer_name'], 'content': review['comments']} for review in reviews_list]
    return formatted_reviews


class SummarizeBase(BaseTool):
    name = "SummarizeBaseTool"
    description = """
    This tool is useful for summarizing property reviews based on user input or when user wants to get reviews for a property.

    Input to this tool should always follow the format: "listing id"

    Example usage:
    Suppose you want to summarize reviews for a listing id "4567":

    <tool_input>listing id for the property</tool_input>
    """

    def extract_id_from_xml(self, xml_string):

        wrapped_xml_string = f"<root>{xml_string}</root>"

        # Parse the XML string
        root = ET.fromstring(wrapped_xml_string)

        id_element = root.find('id')

        if id_element is not None:
            return id_element.text
        else:
            return None

    def _run(self, tool_input):
        llm = ChatBedrock(
            credentials_profile_name="kamal",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            model_kwargs={"temperature": 0},
            streaming=True,
            region_name="us-east-1"
        )
        listing_id = tool_input
        print("*****************summarize base tool")
        print(f"Listing ID: '{listing_id}'")

        # print(id)
        reviews_df = extract_reviews(listing_id)
        # reviews_df = reviews_df.sort_values(by="date")
        if len(reviews_df) > 100:
            return reviews_df.head(100)
        # print(reviews_df)
        if (len(reviews_df)) > 0:
            review_list = reviews_to_list(reviews_df)
            print(review_list)

        

            prompt = """
                    You will be analyzing a set of Airbnb reviews. Your task is to create a summary of all reviews, highlight positive points, and identify negative points from these reviews. Here are the steps to follow:

                1. First, carefully read and analyze all the reviews provided in the following section:

                <reviews>
                    """ + "\n".join(
                [f"<review><review_by>{review['name']}</review_by><content>{review['content']}</content></review>\n" for review in review_list if review['content']]
            ) + """
                </reviews>

                2. After analyzing all the reviews, create a concise summary that captures the overall sentiment and main points mentioned across all reviews. This summary should be a paragraph or two in length.

                3. Next, identify the positive points mentioned in the reviews. These should be specific aspects that guests frequently praised or highlighted as strengths.

                4. Then, identify the negative points or areas of improvement mentioned in the reviews. These should be specific aspects that guests frequently criticized or suggested could be better.

                Present your analysis in the following format, using the specified XML tags:
                
                SUMMARY:
                [Insert your overall summary of the reviews here]
                

                Positive Points:
                - [First positive point] : [name of the guest]
                - [Second positive point] : [name of the guest]
                - [Third positive point] : [name of the guest]
                [Continue with additional positive points as needed]  : [name of the guest]

                Negative Points:
                - [First negative point] : [name of the guest]
                - [Second negative point] : [name of the guest]
                - [Third negative point] : [name of the guest]
                [Continue with additional negative points as needed] : [name of the guest]

                Remember to be objective in your analysis, basing your summary and points strictly on the content of the reviews provided. Ensure that your positive and negative points are clear, concise, and reflect the most commonly mentioned aspects in the reviews.
                """
            messages = [
                {
                    "content": prompt,
                    "role": "user",
                },
            ]   
            print("calling llm")
            response = llm.invoke(prompt).content

            print("got review summary " + response)
            return response
        else:
            return f"Sorry, Didn't find reviews in our database for listing id {listing_id}"


summarize_tool = Tool(
    name=SummarizeBase().name,
    description=SummarizeBase().description,
    func=SummarizeBase().run
)
