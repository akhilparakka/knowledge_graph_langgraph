from minio import Minio

class MinioClient:
    def __init__(self, endpoint, access_key, secret_key):
        self.endpoint = endpoint
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    def create_bucket(self, bucket_name):
        self.client.make_bucket(bucket_name)

    def upload_file(self, bucket_name, object_name, file_path):
        self.client.fput_object(bucket_name, object_name, file_path)

    def download_file(self, bucket_name, object_name, file_path):
        self.client.fget_object(bucket_name, object_name, file_path)

    def list_objects(self, bucket_name):
        return self.client.list_objects(bucket_name)

    def delete_object(self, bucket_name, object_name):
        self.client.remove_object(bucket_name, object_name)

    def delete_bucket(self, bucket_name):
        self.client.remove_bucket(bucket_name)

    def delete_all_objects(self, bucket_name):
        for obj in self.list_objects(bucket_name):
            self.delete_object(bucket_name, obj.object_name)

    def delete_all_buckets(self):
        for bucket in self.client.list_buckets():
            self.delete_bucket(bucket.name)

    def delete_all_objects_in_all_buckets(self):
        for bucket in self.client.list_buckets():
            self.delete_all_objects(bucket.name)
