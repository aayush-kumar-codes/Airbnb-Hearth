from langchain.tools import Tool, BaseTool
from dotenv import load_dotenv
import pandas as pd
from langchain_aws import ChatBedrock

load_dotenv()

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


class Comparebase(BaseTool):
    name = "CompareTool"
    description = """
    useful when user wants to compare reviews between two properties for a specific keyword

    Input to this tool will be always "listing_id_1","listing_id_2","keyword"

    Example: 
    can you compare reviews between these to listings "listing_id", "listing_id" and for "noise levels"

    <tool_input>listing_id_1 id of the first listing, listing_id_2 id of the second listing, keyword to compare reviews on</tool_input>
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

        print("**************knowlegebase tool")
        print("tool input", tool_input, tool_input.split(','))
        splits = tool_input.split(',')
        listing_id_1 = splits[0]
        listing_id_2 = splits[1]
        keyword = splits[2]
        print(f"Parsed input - Prompt: listing_id_1: {listing_id_1}  listing_id_2 {listing_id_2} query {keyword}")

        # print(id)
        reviews_df_1 = extract_reviews(listing_id_1)
        reviews_df_2 = extract_reviews(listing_id_2)
        # print(reviews_df)
        if (len(reviews_df_1)) > 0 and (len(reviews_df_2)) > 0:
            review_list_1 = reviews_to_list(reviews_df_1)
            review_list_2 = reviews_to_list(reviews_df_2)

            prompt = """
                    You will be analyzing a set of Airbnb reviews. 

                    You are given reviews of two properies. You need to compare pros/cons of reviews for both properties for a specific keyword.

                    Your task is to highlight positive points, and identify negative points from these reviews. Here are the steps to follow:

                1. First, carefully read and analyze all the reviews provided in the following section:

                <reviews_property_1>
                    """ + "\n".join(
                [f"<review><review_by>{review['name']}</review_by><content>{review['content']}</content></review>\n" for review in review_list_1 if review['content']]
            ) + f"""
                </reviews_property_1>

                <reviews_property_2>
                    """ + "\n".join(
                [f"<review><review_by>{review['name']}</review_by><content>{review['content']}</content></review>\n" for review in review_list_2 if review['content']]
            ) + f"""
                </reviews_property_2>

                2. Extract and analysize reviews only related to keyword "{keyword}" for both properties

                2. After analyzing all the reviews related to keyword "{keyword}", create a concise summary that captures the overall sentiment and main points.
                This summary should be a paragraph or two in length.

                Present your analysis in the following format

                Comparison of reviews between properity1 and property2 related to {keyword}
                - comparison1
                - comparison2
                - so on

                If {keyword} doesn't exist in reviews, mention the review's doesn't existing for {keyword}

                Remember to be objective in your analysis, basing your summary and points strictly on the content of the reviews provided.
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
            return f"Sorry, Didn't find reviews in our database for listing id {listing_id_1} or {listing_id_2}"


compare_tool = Tool(
    name=Comparebase().name,
    description=Comparebase().description,
    func=Comparebase().run
)
