#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Aseguramos que las carpetas montadas existan y sean escribibles por el
# usuario 'algorithm' del contenedor (el bind-mount hereda los permisos del host).
mkdir -p "$SCRIPTPATH/test/output" "$SCRIPTPATH/test/cache"
chmod -R 777 "$SCRIPTPATH/test/output" "$SCRIPTPATH/test/cache"

# Nota: NO usamos --user root. El contenedor ya corre como 'algorithm' (USER del
# Dockerfile), que es dueño de /opt/algorithm. Con --cap-drop=ALL, root perdería
# CAP_DAC_OVERRIDE y no podría escribir en carpetas de 'algorithm'.
docker run --rm \
        --memory="30g" \
        --memory-swap="30g" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="2g" \
        -v "$SCRIPTPATH/test/input/:/input/:ro" \
        -v "$SCRIPTPATH/test/output/:/output/" \
        -v "$SCRIPTPATH/test/cache/:/cache/" \
        autopet_baseline