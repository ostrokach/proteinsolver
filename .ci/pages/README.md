# Pages

This file creates a `./public` folder containing documentation created for multiple versions (tags) of this repository.

When the repository is public, our job is easy: we simply download the `artifact.zip` file from a publicly-accessible URL (see: [downloading the latest artifacts]). However, when the repository is private, using the above-mentioned URL does not work (see: [gitlab-org/gitlab-ce#22957]). In that case, we resort to using the GitLab API instead.

If [gitlab-org/gitlab-ce#22957] is ever fixed, we would be able to specify
`--header "Private-Token: XXXXX"` or attach `&private_token=XXXXX` to the query string,
and keep using the original URL:

```bash
curl --header "Private-Token: XXXXX" \
    "https://gitlab.com/user/repo/-/jobs/artifacts/ref/download?job=job_name"
```

Good resource: <https://docs.gitlab.com/ee/api/jobs.html#download-the-artifacts-archive>.

<!-- Links -->

[downloading the latest artifacts]: https://docs.gitlab.com/ee/user/project/pipelines/job_artifacts.html#downloading-the-latest-artifacts
[gitlab-org/gitlab-ce#22957]: https://gitlab.com/gitlab-org/gitlab-ce/issues/22957
