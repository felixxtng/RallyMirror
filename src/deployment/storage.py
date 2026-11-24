"""Private object storage. Only UUID-derived keys are used, never user paths."""
from pathlib import Path
from botocore.config import Config
import boto3

def key(job_id, name):
    import uuid
    if str(uuid.UUID(job_id)) != job_id:
        raise ValueError("Invalid job id")
    if name not in {"upload.mp4", "input.mp4", "result.mp4", "report.json"}:
        raise ValueError("Invalid artifact")
    return f"jobs/{job_id}/{name}"

class Storage:
    def __init__(self, settings):
        self.settings = settings
        self.client = None
        if settings.backend == "runpod":
            self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint,
                region_name="auto", config=Config(signature_version="s3v4"))

    def path(self, job_id, name):
        return self.settings.root / key(job_id, name)

    def size(self, job_id, name):
        if self.client:
            return self.client.head_object(Bucket=self.settings.bucket, Key=key(job_id, name))["ContentLength"]
        return self.path(job_id, name).stat().st_size

    def url(self, job_id, name, upload=False):
        params = {"Bucket": self.settings.bucket, "Key": key(job_id, name)}
        if upload:
            params["ContentType"] = "video/mp4"
        return self.client.generate_presigned_url("put_object" if upload else "get_object",
            Params=params, ExpiresIn=900)

    def download(self, job_id, name, destination):
        self.client.download_file(self.settings.bucket, key(job_id, name), str(destination))

    def upload(self, job_id, name, source):
        self.client.upload_file(str(source), self.settings.bucket, key(job_id, name),
            ExtraArgs={"ContentType": "video/mp4" if name.endswith("mp4") else "application/json"})

    def freeze_input(self, job_id):
        # Browser can overwrite its temporary upload URL until expiry. The
        # worker reads a separate key never signed for browser writes.
        self.client.copy_object(Bucket=self.settings.bucket, Key=key(job_id, "input.mp4"),
            CopySource={"Bucket": self.settings.bucket, "Key": key(job_id, "upload.mp4")})
