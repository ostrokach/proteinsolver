#!/bin/bash

set -ev

# Make sure required parameters have been set
REQUIRED_VARS=(CI_PROJECT_ID CI_DOCS_TOKEN OUTPUT_DIR)
for var in ${REQUIRED_VARS[*]} ; do
    if [[ -z "${!var}" ]] ; then
        echo "Environment variable '${var}' has not been set!"
        exit -1
    fi
done

# Parameters
TEMP_DIR=temp

# Create a folder for temporary work
mkdir -p ${TEMP_DIR}
pushd ${TEMP_DIR}

# Create a list of tags
git tag -l --sort="-v:refname" | tee tags.txt

# Remove tags that we wish to ignore
if [[ "x${TAGS_TO_IGNORE}" != "x" ]] ; then
    rg -v "${TAGS_TO_IGNORE}" tags.txt > tags_filtered.txt ;
    mv tags_filtered.txt tags.txt ;
fi

# Download docs artifacts and rename the public folder to ${OUTPUT_DIR}/${tag}
while read tag ; do
    echo "Downloading artifacts for ${tag}..."
    curl --header "PRIVATE-TOKEN: $CI_DOCS_TOKEN" -L -s -o artifact.zip \
        https://gitlab.com/api/v4/projects/${CI_PROJECT_ID}/jobs/artifacts/${tag}/download?job=docs
    file artifact.zip
    unzip -o -q artifact.zip || true
    mv -f public "${OUTPUT_DIR}/${tag}" || true
done <tags.txt

# Clean up temporary files
popd
rm -rf ${TEMP_DIR}
