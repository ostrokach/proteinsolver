import os

#: Location of training and validation data
data_url = os.getenv("DATAPKG_DATA_DIR", "https://storage.googleapis.com")

#: Whether `data_url` refers to a remote location.
data_is_remote = not (data_url.startswith("file://") or data_url.startswith("/"))
