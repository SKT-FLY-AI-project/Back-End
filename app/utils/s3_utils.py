import boto3
from app.config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET_NAME, S3_REGION

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def upload_to_s3(userid: str, file_path: str, file_name: str):
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, file_name)
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{file_name}"
        return s3_url
    except Exception as e:
        print("S3 Upload Error:", e)
        return None
