import argparse
import concurrent.futures
import functools
import itertools
import logging
import os
import re
import shutil
import string
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Union

import gitlab
from jinja2 import Template

logger = logging.getLogger()
PROJECT_ROOT = Path(__file__).parent


class JobInfo(NamedTuple):
    job_id: int
    tag_name: str
    finished_at: datetime
    folder: Optional[Path] = None


def main(args):
    """
    Args:
        job_name: Name of the job building desired artifacts.
        project_id: ID of the project.
        private_token: Token for accessing GitLab API.
        output_path: Location where pages should be extracted
    """
    output_path = Path(args.output_dir).expanduser().resolve(strict=True)

    existing_jobs = [
        JobInfo(0, p.name, datetime.fromtimestamp(p.stat().st_ctime), p)
        for p in output_path.glob("*")
        if p.is_dir()
    ]
    existing_tag_names = {j.tag_name for j in existing_jobs}
    logger.info("Existing tag names: %s.", existing_tag_names)

    if args.project_id is not None:
        previous_jobs = download_pervious_versions(
            args.project_id, args.job_name, args.private_token, output_path, existing_tag_names
        )
    else:
        previous_jobs = []

    all_jobs = sort_jobs_by_tag(existing_jobs + previous_jobs)
    index_source = render_html(all_jobs)
    write_index_files(index_source, output_path)


def download_pervious_versions(
    project_id: Union[str, int],
    job_name: str,
    private_token: str,
    output_path: Path,
    existing_tag_names: List[str] = [],
):
    gl = gitlab.Gitlab("https://gitlab.com", private_token=private_token)
    project = gl.projects.get(project_id)
    refs = {t.name for t in project.tags.list(all=True, as_list=False)}

    job_list = get_job_list(project, job_name=job_name, refs=refs)
    job_list = remove_duplicate_tags(job_list)
    job_list = [j for j in job_list if j.tag_name not in existing_tag_names]

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_files = download_artifacts(project, job_list, Path(temp_dir))
        job_paths = extract_artifacts(job_list, artifact_files, output_path)
    job_list = [j._replace(folder=jp) for j, jp in zip(job_list, job_paths)]
    return job_list


def get_job_list(project, job_name=None, refs=None) -> List[JobInfo]:
    job_list: List[JobInfo] = []
    for pipeline in project.pipelines.list(all=True, as_list=False):
        if refs is None or pipeline.attributes["ref"] in refs:
            for job in pipeline.jobs.list(all=True, as_list=False):
                if job.attributes["status"] == "success" and (
                    job_name is None or job.attributes["name"] == job_name
                ):
                    finished_at = datetime.strptime(
                        job.attributes["finished_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    job_list.append(JobInfo(job.id, job.attributes["ref"], finished_at))
    return job_list


def remove_duplicate_tags(job_list: List[JobInfo]) -> List[JobInfo]:
    job_list = sort_jobs_by_date(job_list)

    _seen: Set[str] = set()
    job_list = [
        j for j in job_list if j.tag_name not in _seen and not _seen.add(j.tag_name)  # type: ignore
    ]
    return job_list


def sort_jobs_by_date(job_list: List[JobInfo]) -> List[JobInfo]:
    job_list = sorted(job_list, key=lambda x: x.finished_at, reverse=True)
    assert len(job_list) == 0 or job_list[0].finished_at >= job_list[-1].finished_at
    return job_list


def sort_jobs_by_tag(job_list: List[JobInfo]) -> List[JobInfo]:
    def _str_to_float(s):
        s = s.strip(string.ascii_letters)
        try:
            return float(s)
        except ValueError:
            return float("inf")

    job_list = sorted(
        job_list,
        key=lambda j: tuple(_str_to_float(s) for s in re.split(r"\s|\.|-|_", j.tag_name)),
        reverse=True,
    )

    return job_list


def download_artifacts(project, job_list: List[JobInfo], temp_dir: Path) -> List[Path]:
    artifact_files = []
    for j in job_list:
        job = project.jobs.get(j.job_id, lazy=True)
        artifact_file = Path(temp_dir, j.tag_name + ".zip")
        with artifact_file.open("wb") as fout:
            job.artifacts(streamed=True, action=fout.write)
        artifact_files.append(artifact_file)
    return artifact_files


def extract_artifacts(job_list: List[JobInfo], artifact_files: List[Path], output_path: Path):
    """
    This may be faster than using :any:`zipfile.ZipFile`.
    """
    unzip = shutil.which("unzip")
    Pool: object
    if unzip is not None:
        Pool = concurrent.futures.ThreadPoolExecutor
        fn = functools.partial(_extract_artifact, _extract_fn=_extract1)
    else:
        Pool = concurrent.futures.ProcessPoolExecutor
        fn = functools.partial(_extract_artifact, _extract_fn=_extract2)
    with Pool() as pool:
        futures = pool.map(fn, job_list, artifact_files, itertools.repeat(output_path))
        results = list(futures)
    return results


def _extract1(artifact_file: Path, temp_dir: str):
    proc = subprocess.run(
        ["unzip", "-o", "-q", "-d", temp_dir, artifact_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    output = proc.stdout + proc.stderr
    if output.strip():
        logger.info(output.strip())


def _extract2(artifact_file: Path, temp_dir: str):
    with zipfile.ZipFile(artifact_file, "r") as zip_file:
        zip_file.extractall(temp_dir)


def _extract_artifact(
    job_info: JobInfo, artifact_file: Path, output_path: Path, _extract_fn=_extract1
) -> Optional[Path]:
    """
    Args:
        artifact_file: Zip archive containing artifact data. Should have name '{tag}.zip'.
        output_path: Location where the 'public' artifact folder will be extracted.
    """
    version_path = Path(output_path, artifact_file.stem)
    if version_path.exists():
        msg = f"Version path '{version_path}' already exists! Skipping."
        logger.warning(msg)
        return None
    with tempfile.TemporaryDirectory() as temp_dir:
        _extract_fn(artifact_file, temp_dir)
        folders = os.listdir(temp_dir)
        if "public" not in folders:
            msg = f"The artifact archive should contain a 'public' folder! Found {folders}."
            logger.info(msg)
            return None
        else:
            shutil.move(Path(temp_dir, "public"), version_path)
            finished_at_seconds = int(job_info.finished_at.strftime("%s"))
            os.utime(version_path, (finished_at_seconds, finished_at_seconds))
            return version_path


def render_html(job_list: List[JobInfo]) -> str:
    with PROJECT_ROOT.joinpath("templates", "index.html").open("rt") as fin:
        template_src = fin.read()
    job_list = [j for j in job_list if j.folder is not None]
    template = Template(template_src)
    source = template.render(items=job_list)
    return source


def write_index_files(index_source: str, output_path: Path):
    with output_path.joinpath("index.html").open("wt") as fout:
        fout.write(index_source)
    for path in PROJECT_ROOT.joinpath("static").glob("*"):
        shutil.copy2(path, output_path.joinpath(path.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--private-token", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    sys.exit(main(args))
