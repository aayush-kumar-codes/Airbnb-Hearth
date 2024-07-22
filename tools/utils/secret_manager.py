import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
load_dotenv()

SECRET_NAME = os.getenv("SECRET_NAME")
REGION_NAME = "us-east-1"


def get_secrets(key_to_retrieve):
    # Create a Secrets Manager client
    session = boto3.session.Session(profile_name='kamal',
                                    aws_access_key_id=os.getenv("aws_access_key"),
                                    aws_secret_access_key=os.getenv("aws_secret_key"),
                                    region_name=REGION_NAME)
    client = session.client(
        service_name='secretsmanager',
        region_name=REGION_NAME
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=SECRET_NAME
        )

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"The requested secret {SECRET_NAME} was not found")
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            print(f"The request was invalid due to: {e}")
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            print(f"The request had invalid params: {e}")
        else:
            raise e
    else:
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            secret = eval(secret)  # Convert string to dictionary
            print(secret)
            result = secret[key_to_retrieve]
            return result
