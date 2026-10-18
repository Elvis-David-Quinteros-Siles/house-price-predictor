#!/bin/sh
# Genera los PNG de docs/MLOPS.md a partir de las fuentes Mermaid (*.mmd).
# GitHub no siempre dibuja los bloques Mermaid de un documento con muchos
# diagramas, así que el documento enlaza imágenes. Tras editar un .mmd:
#   sh docs/diagramas/render.sh
set -eu
cd "$(dirname "$0")"
IMAGE=minlag/mermaid-cli:11.4.2
for source in *.mmd; do
    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/data" "$IMAGE" \
        -i "/data/$source" -o "/data/${source%.mmd}.png" --theme default --width 1400 --scale 2 --backgroundColor white --quiet
    echo "${source%.mmd}.png"
done
