#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo "Building Docker image..."
docker build -t autopet_baseline "$SCRIPTPATH"