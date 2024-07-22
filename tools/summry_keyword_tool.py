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

    listing_id = int(listing_id)

    filtered_df = df[df['listing_id'] == listing_id]

    reviews = filtered_df[['reviewer_name', 'comments']]

    return reviews


def reviews_to_list(reviews_df):
    reviews_list = reviews_df.to_dict(orient='records')
    formatted_reviews = [{'name': review['reviewer_name'], 'content': review['comments']} for review in reviews_list]
    return formatted_reviews


class ReviewHighlightBase(BaseTool):
    name = "ReviewHighlightTool"
    description = """
    This tool is useful for summarizing reviews for a property or list based on a specific keyword user wants to search in reviews.

    Input to this tool should always follow the format: "listing_id (integer), keyword (string)"

    Example usage:
    Can you highlight reviews related to "child friendliness" listing id "4567":

    <tool_input>listing id for the property in integer format, specific review user wants to search in reviews in string</tool_input>
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
        splits = tool_input.split(',')
        listing_id = splits[0]
        keyword = " , ".join(splits[1:])
        print("*****************summarize keyword base tool")
        print(f"Listing ID: '{listing_id}' keyword {keyword}")

        # print(id)
        reviews_df = extract_reviews(listing_id)
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
            ) + f"""
                </reviews>

                2. Extract and analysize reviews only related to keyword "{keyword}"

                2. After analyzing all the reviews related to keyword "{keyword}", create a concise summary that captures the overall sentiment and main points mentioned across all reviews. This summary should be a paragraph or two in length.

                3. Next, identify the positive points mentioned in the reviews related to keyword "{keyword}". These should be specific aspects that guests frequently praised or highlighted as strengths.

                4. Then, identify the negative points or areas of improvement mentioned in the reviews related to keyword "{keyword}". These should be specific aspects that guests frequently criticized or suggested could be better.

                Present your analysis in the following format, using the specified XML tags:

                SUMMARY:
                [Insert your overall summary of the reviews here related to keyword "{keyword}"]


                Positive Points:
                - [First positive point related to keyword "{keyword}"] : [name of the guest]
                - [Second positive point related to keyword "{keyword}"] : [name of the guest]
                - [Third positive point related to keyword "{keyword}"] : [name of the guest]
                [Continue with additional positive points as needed]  : [name of the guest]

                Negative Points:
                - [First negative point related to keyword "{keyword}"] : [name of the guest]
                - [Second negative point related to keyword "{keyword}"] : [name of the guest]
                - [Third negative point related to keyword "{keyword}"] : [name of the guest]
                [Continue with additional negative points as needed] : [name of the guest]

                Remember to be objective in your analysis, basing your summary and points strictly on the content of the reviews provided. Ensure that your positive and negative points are clear, concise, and reflect the most commonly mentioned aspects in the reviews.
                Make sure analysis is related to keyword "{keyword}"
                """
            messages = [
                {
                    "content": prompt,
                    "role": "user",
                },
            ]

            response = llm.invoke(prompt).content

            print("got review summary " + response)
            return response
        else:
            return f"Sorry, Didn't find reviews in our database for listing id {listing_id}"


highlight_tool = Tool(
    name=ReviewHighlightBase().name,
    description=ReviewHighlightBase().description,
    func=ReviewHighlightBase().run
)
