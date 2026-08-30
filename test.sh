#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Make the mounted output/cache writable by the 'algorithm' user in the container.
mkdir -p "$SCRIPTPATH/test/output" "$SCRIPTPATH/test/cache"
chmod -R 777 "$SCRIPTPATH/test/output" "$SCRIPTPATH/test/cache"

# Local CPU-only run (no --gpus). process.py auto-detects and falls back to CPU.
docker run --rm \
        --cpus="8" \
        --memory="30g" \
        --memory-swap="30g" \
        --shm-size="2g" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        -v "$SCRIPTPATH/test/input/:/input/:ro" \
        -v "$SCRIPTPATH/test/output/:/output/" \
        -v "$SCRIPTPATH/test/cache/:/cache/" \
        uam_team_submission