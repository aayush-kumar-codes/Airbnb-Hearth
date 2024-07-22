import ast
# from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import os
import xml.etree.ElementTree as ET
import re
import boto3
from langchain.schema import Document
from langchain_community.embeddings import BedrockEmbeddings
# from opensearchpy import RequestsHttpConnection
# from requests_aws4auth import AWS4Auth
# from langchain_community.vectorstores import OpenSearchVectorSearch
# from utils.secret_manager import get_secrets
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

columns = [
    'id',
    'name',
    'description',
    'property_type',
    'room_type',
    'accommodates',
    'bathrooms_text',
    'bedrooms',
    'amenities',
    'price',
    'review_scores_rating',
    'host_location',
    'neighborhood_overview',
    'host_neighbourhood'
]


client = boto3.client("s3")

# state_file_extension_s3 = ['NY']

# for file_extension_state in state_file_extension_s3:

print(f"Processing the listings.csv file")

df = pd.read_csv('listings.csv')[columns]

# print(f"Processing the {file_extension_state} file")

# df = pd.read_csv(f'{file_extension_state}.csv')[columns]

df_ny = df[df['host_location'] == 'New York, NY']

df2 = pd.read_csv("reviews.csv")
print(df2.columns)
print(len(df2))

# merged_df = pd.merge(df, df2, left_on='id', right_on='listing_id')
# merged_df = merged_df.dropna(subset=['comments'])  # Adjust the column name as per your dataset
# merged_df = merged_df.drop(columns=['listing_id'])

ids_with_reviews = df2['listing_id'].unique()
filtered_df = df[df['id'].isin(ids_with_reviews)]


print(len(df))
print(len(filtered_df))

df = filtered_df

def parse_and_join(s):
    try:
        # Convert string representation of list to actual list
        lst = ast.literal_eval(s)
        # Join list elements into a single string
        return ','.join(str(item) for item in lst)
    except:
        None


df_ny.rename(columns={'name': 'property_name'}, inplace=True)
df_ny.rename(columns={'description': 'property_description'}, inplace=True)
# df['amenities'] = df['amenities'].apply(parse_and_join)
# df['review_scores_rating'] = df['review_scores_rating'].fillna(0)


df_ny['amenities'] = df_ny['amenities'].apply(parse_and_join)
df_ny['review_scores_rating'] = df_ny['review_scores_rating'].fillna(0)


def clean_text(text):
    # Convert to string if not already
    text = str(text)

    # Remove .<br /><br /> and &lt;br /&gt;&lt;br /&gt;
    cleaned = re.sub(r'\.?(<br\s*/?>){2}|\.?(&lt;br\s*/&gt;){2}', '', text)

    # Remove any remaining HTML tags
    cleaned = re.sub(r'<[^>]+>|&lt;[^&]+&gt;', '', cleaned)

    return cleaned.strip()


def row_to_xml(row):
    xml_elements = []
    for column, value in row.items():
        if column != 'host_location':
            elem = ET.Element(column)
            elem.text = clean_text(value)
            xml_elements.append(ET.tostring(elem, encoding='unicode', method='xml'))
    return ' '.join(xml_elements)  # Join with two spaces


xml_strings = []

print(len(df_ny))
df_ny = df_ny.dropna(subset=['price'])
df_ny = df_ny.dropna(subset=['review_scores_rating'])
df_ny = df_ny.dropna(subset=['amenities'])
df_ny = df_ny.dropna(subset=['neighborhood_overview'])
print(len(df_ny))
    

record_count = 0
for _, row in df_ny.iterrows():

    # print(row)
    # print(row_to_xml(row))
    

    xml_strings.append(Document(row_to_xml(row), metadata={"state": "NY", "id": row["id"]}))
    record_count += 1
    if record_count >= 10000:
        break


print(f"loading the {record_count} records")

region = 'us-east-1'
bedrock_client = boto3.client("bedrock-runtime", aws_access_key_id=os.getenv("aws_access_key"),
                              aws_secret_access_key=os.getenv("aws_secret_key"),
                              region_name=region)
embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

# for doc in xml_strings:
#     print(doc)
#     break

vectorstore_faiss = FAISS.from_documents([xml_strings[0]], embedding)

for doc in tqdm(xml_strings[1:], desc="Adding documents"):
    vectorstore_faiss.add_documents([doc])

# vectorstore_faiss = FAISS.from_documents(
#     xml_strings,
#     embedding,
# )

vectorstore_faiss.save_local("./data")

# awsauth = AWS4Auth(get_secrets(os.getenv("open_search_access_key")), get_secrets(os.getenv("open_search_secret_key")), 'us-east-1', 'aoss',
#                     session_token=None)
# #
# vectordb = OpenSearchVectorSearch.from_documents(
#     documents=xml_strings,
#     embedding=embedding,
#     opensearch_url=get_secrets(os.getenv("open_search_url")),
#     http_auth=awsauth,
#     index_name="airbnb_100_index",
#     timeout=300,
#     use_ssl=True,
#     verify_certs=True,
#     connection_class=RequestsHttpConnection,
#     engine="faiss",
#     bulk_size=5000
# )


print(f"Completed processing")
