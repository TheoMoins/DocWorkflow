#!/bin/bash

set -e

PIPELINE_DIR="configs/pipeline"

for config in ${PIPELINE_DIR}/*.yml; do
    run_name=$(grep '^run_name:' "$config" | sed 's/run_name: *"\?\([^"]*\)"\?/\1/')
    output_dir=$(grep '^output_dir:' "$config" | sed 's/output_dir: *"\?\([^"]*\)"\?/\1/')
    result_dir="${output_dir}/${run_name}"

    echo "========================================"
    echo "Config: ${config}"
    echo "Result dir: ${result_dir}"
    echo "========================================"

    echo "[predict] ${config}"
    docworkflow -c "${config}" predict -t all

    echo "[score] ${config}"
    docworkflow -c "${config}" score -t htr -p "${result_dir}"

    echo "Done: ${run_name}"
    echo ""
done

echo "All configs processed."
