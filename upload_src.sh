#!/usr/bin/env bash

ARTIFACTS_PATH="tmp/artifacts"

rm -rf "${ARTIFACTS_PATH}"
mkdir -p "${ARTIFACTS_PATH}"

tar --exclude='__pycache__' \
  -czvf "${ARTIFACTS_PATH}/src.tar.gz" "src"
cp dataset-metadata-src.json "${ARTIFACTS_PATH}/dataset-metadata.json"

kaggle datasets version -p "${ARTIFACTS_PATH}" -r tar -m "Updated at $(date)"