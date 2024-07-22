import pandas as pd
df = pd.read_csv('listings.csv')

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

filtered_df['id'] = filtered_df['id'].astype(int)
df2["listing_id"] = df2['listing_id'].astype(int)

print(len(filtered_df[filtered_df['id'] == 544039]))
print(len(df2[df2["listing_id"] == 544039]))

# from claude import callLLMHaikuViaMessages

# print(callLLMHaikuViaMessages("", [{
#     "role": "user",
#     "content": "Hi how are you"
# }]))


# import re
# import boto3
# from langchain.schema import Document
# from langchain_community.embeddings import BedrockEmbeddings
# import os
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()


# region = 'us-east-1'
# bedrock_client = boto3.client("bedrock-runtime", aws_access_key_id=os.getenv("aws_access_key"),
#                               aws_secret_access_key=os.getenv("aws_secret_key"),
#                               region_name=region)
# embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)


# vectorstore_faiss = FAISS.load_local("./data", embedding, allow_dangerous_deserialization=True)

# query = "appartment in airbnb"

# query_embedding = embedding.embed_query(query)
# relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)

# for i, rel_doc in enumerate(relevant_documents):
#     print(f'## Document {i+1}: {rel_doc.metadata} {rel_doc.page_content}')
#     print(rel_doc.metadata["id"])
