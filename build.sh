#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo "Building Docker image..."
# Docker repository names must be lowercase. The submission is named
# "UAM_team_submission" on Grand Challenge; the local image tag is its lowercase form.
docker build -t uam_team_submission "$SCRIPTPATH"