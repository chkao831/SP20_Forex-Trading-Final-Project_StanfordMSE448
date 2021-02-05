'''
Created for Winter 2019 Stanford CS224W
Jingbo Yang, Ruge Zhao, Meixian Zhu
Pytorch-specific implementation
'''


from gcloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os

from constants import *

# Nicely rephrased version of the following
# https://stackoverflow.com/questions/37003862/google-cloud-storage-how-to-upload-a-file-from-python-3

# Useful page describing what to do
# https://hackersandslackers.com/manage-files-in-google-cloud-storage-with-python/


def make_dir_until_last(path):
    chunks = str(path).split('/')
    chunks = '/'.join(chunks[:-1])
    os.makedirs(chunks, exist_ok=True)


def path_to_str(function):
    def wrapper(*args):
        if args[0].fake:
            return lambda x: 'Fake GCloud Storage'

        all_args = [args[0]]
        for a in args[1:]:
            all_args.append(str(a))
        result = function(*all_args)
        return result
    return wrapper


class GCStorage:
    '''Utility class for interaction with Google Cloud storage

        project name = google cloud project id
        credential = JSON file for service account
        bucket name = well, bucket name
    '''
    
    MONO = None

    @staticmethod
    def get_credentials(credential_path):
        return ServiceAccountCredentials.\
                    from_json_keyfile_name(credential_path)

    @staticmethod
    def get_CloudFS(*args, **kwargs):
        if GCStorage.MONO is not None:
            return GCStorage.MONO
        else:
            GCStorage.MONO = GCStorage(*args, **kwargs)
            print('Unique instance for GCStorage has been created')
            return GCStorage.MONO

    def __init__(self, project_name, bucket_name, credential_path, fake=False):
        self.client = storage.Client(
            credentials=GCStorage.get_credentials(credential_path),
            project=project_name
            )
        self.bucket = self.client.get_bucket(bucket_name)
        self.bucket_name = bucket_name
        self.fake = fake
    
    def get_bucket_root(self):
        return f'gs://{self.bucket_name}/'

    @path_to_str
    def upload(self, local_path, cloud_path):
        '''Upload file to GCP bucket'''
        blob = self.bucket.blob(cloud_path)
        blob.upload_from_filename(local_path)
        string = f'Uploaded {local_path} to "{cloud_path}" bucket.'
        # print(string)
        return string

    @path_to_str
    def download(self, local_path, cloud_path):
        '''Download file from GCP bucket'''
        make_dir_until_last(local_path)

        blob = self.bucket.blob(cloud_path)
        blob.download_to_filename(local_path)
        return f'{cloud_path} downloaded from bucket.'

    @path_to_str
    def list_files(self, storge_path, delimiter='/'):
        '''List all files in GCP bucket'''
        files = self.bucket.list_blobs(prefix=storge_path)
        # file_list = [f.name for f in files]

        directory = dict()
        for f in files:
            name = f.name
            levels = name.split(delimiter)
            cur_dir = directory
            for l in levels:
                if l not in cur_dir:
                    cur_dir[l] = dict()
                cur_dir = cur_dir[l]
        
        try:
            cur_dir = directory
            for l in storge_path.split(delimiter):
                cur_dir = cur_dir[l]
        except:
            print(f'storge_path {storge_path} does not exist')
            cur_dir = []

        return directory, cur_dir

    @path_to_str
    def delete(self, cloud_path):
        '''Delete file from GCP bucket'''
        self.bucket.delete_blob(cloud_path)
        return f'{cloud_path} deleted from bucket.'

    @path_to_str
    def rename(self, cloud_path_orig, cloud_path_new):
        '''Rename file in GCP bucket'''
        blob = self.bucket.blob(cloud_path_orig)
        self.bucket.rename_blob(blob, new_name=cloud_path_new)
        return f'{cloud_path_orig} renamed to {cloud_path_new}.'


class GCOpen():
    '''Custom file writer'''

    def __init__(self, cloud_path, file_mode, gc=None,
                 temp_path=TEMP_FOLDER, use_cloud=True):
        self.cloud_path = cloud_path
        self.file_mode = file_mode
        self.use_cloud = use_cloud
        
        if gc is not None:
            self.gc = gc
        else:
            self.gc = GCStorage.get_CloudFS()

        filename = str(cloud_path).split('/')[-1]
        self.temp_path = temp_path / filename

    def open(self):
        if 'w' in self.file_mode:
            pass
        else:
            if self.use_cloud:
                self.gc.download(self.temp_path, self.cloud_path)
            else:
                pass

        self.file = open(self.temp_path, mode=self.file_mode)
        return self.file

    def __enter__(self):
        return self.open() 

    def __exit__(self, type, value, traceback):
        self.close()

    def send_to_cloud(self):
        if self.use_cloud:
            if 'w' in self.file_mode:
                self.gc.upload(self.temp_path, self.cloud_path)
            elif 'r' in self.file_mode:
                pass

    def flush(self):
        self.file.flush()
        self.send_to_cloud()

    def close(self):
        self.file.close()
        self.send_to_cloud()
