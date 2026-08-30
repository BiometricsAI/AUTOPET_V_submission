#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Rebuild first so the exported image is always up to date.
bash "$SCRIPTPATH/build.sh"

echo "Exporting image to UAM_team_submission.tar.gz ..."
# Docker repo name is lowercase (uam_team_submission); the uploaded file keeps
# the submission name "UAM_team_submission".
docker save uam_team_submission | gzip -c > "$SCRIPTPATH/UAM_team_submission.tar.gz"

echo "Done: $SCRIPTPATH/UAM_team_submission.tar.gz"