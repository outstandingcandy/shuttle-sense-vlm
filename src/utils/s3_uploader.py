#!/usr/bin/env python3
"""
S3 Uploader Module - Handles uploading video files to AWS S3 and generating public URLs
"""

import os
import logging
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class S3VideoUploader:
    """Handles video file uploads to AWS S3 with public URL generation"""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        s3_prefix: str = "videos/"
    ):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: S3 bucket name (defaults to S3_BUCKET_NAME env var)
            aws_region: AWS region (defaults to AWS_REGION env var or 'us-west-2')
            aws_access_key_id: AWS access key ID (defaults to AWS_ACCESS_KEY_ID env var)
            aws_secret_access_key: AWS secret access key (defaults to AWS_SECRET_ACCESS_KEY env var)
            s3_prefix: Prefix for S3 object keys (default: "videos/")
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-west-2")
        self.s3_prefix = s3_prefix

        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided or set in S3_BUCKET_NAME environment variable")

        # Initialize S3 client
        session_kwargs = {"region_name": self.aws_region}

        # Use explicit credentials if provided, otherwise fall back to default credential chain
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
        elif aws_access_key_id or aws_secret_access_key:
            logger.warning("Both aws_access_key_id and aws_secret_access_key must be provided together")

        try:
            self.s3_client = boto3.client("s3", **session_kwargs)
            logger.info(f"Initialized S3 client for bucket: {self.bucket_name} in region: {self.aws_region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise

    def upload_video(
        self,
        local_path: str,
        s3_key: Optional[str] = None,
        expiration: int = 3600,
        content_type: str = "video/mp4"
    ) -> str:
        """
        Upload a video file to S3 and return a presigned URL.

        Args:
            local_path: Path to local video file
            s3_key: S3 object key (defaults to s3_prefix + filename)
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)
            content_type: MIME type of the video file

        Returns:
            Presigned URL of the uploaded video

        Raises:
            FileNotFoundError: If local file doesn't exist
            ClientError: If upload fails
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Video file not found: {local_path}")

        # Generate S3 key if not provided
        if s3_key is None:
            s3_key = f"{self.s3_prefix}{local_path.name}"

        try:
            # Prepare upload parameters (no public ACL)
            extra_args = {"ContentType": content_type}

            # Upload file
            logger.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                str(local_path),
                self.bucket_name,
                s3_key,
                ExtraArgs=extra_args
            )

            # Generate presigned URL
            url = self.generate_presigned_url(s3_key, expiration)
            logger.info(f"Successfully uploaded video and generated presigned URL (expires in {expiration}s)")

            return url

        except ClientError as e:
            logger.error(f"Failed to upload video to S3: {e}")
            raise

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL for an S3 object.

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            Presigned URL that grants temporary access to the object
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise

    def delete_video(self, s3_key: str) -> bool:
        """
        Delete a video from S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Deleting s3://{self.bucket_name}/{s3_key}")
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete video from S3: {e}")
            return False

    def check_if_s3_url(self, url_or_path: str) -> bool:
        """
        Check if a given string is an S3 URL.

        Args:
            url_or_path: URL or file path to check

        Returns:
            True if it's an S3 URL, False otherwise
        """
        try:
            parsed = urlparse(url_or_path)
            return parsed.scheme in ["s3", "https"] and "s3" in parsed.netloc
        except Exception:
            return False

    def upload_if_local(self, video_path: str, cleanup: bool = False, expiration: int = 3600) -> str:
        """
        Upload video to S3 if it's a local file, otherwise return the URL as-is.

        Args:
            video_path: Local path or URL to video
            cleanup: Whether to delete the local file after upload
            expiration: URL expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            S3 presigned URL if uploaded, original URL if already remote
        """
        # If it's already an S3 URL, return as-is
        if self.check_if_s3_url(video_path):
            logger.info(f"Video is already an S3 URL: {video_path}")
            return video_path

        # If it's a local file, upload it
        if os.path.exists(video_path):
            s3_url = self.upload_video(video_path, expiration=expiration)

            if cleanup:
                try:
                    os.remove(video_path)
                    logger.info(f"Deleted local file: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete local file: {e}")

            return s3_url

        # Otherwise, assume it's already a remote URL
        logger.info(f"Assuming video is a remote URL: {video_path}")
        return video_path
