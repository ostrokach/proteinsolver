import shutil
from pathlib import Path
from urllib.request import urlopen


def download_url(url, folder):
    filename = url.rsplit("/", 1)[-1]
    folder = Path(folder)
    folder.mkdir(exist_ok=True)

    if url.startswith("file://") or url.startswith("/"):
        shutil.copy(url.replace("file://", ""), folder)
    else:
        chunk_size = 16 * 1024
        response = urlopen(url)
        with (folder / filename).open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
