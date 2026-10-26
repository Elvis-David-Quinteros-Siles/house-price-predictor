#!/usr/bin/env bash
# Funciones comunes para los scripts de Vault de house-price-predictor
set -euo pipefail
VAULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$VAULT_DIR")"
PROJECT="house-price-predictor"
VAULT_IMAGE="hashicorp/vault:${VAULT_IMAGE_TAG:-2.1.1}"
VAULT_NETWORK="${VAULT_NETWORK:-proxy}"
VAULT_ADDR_INTERNAL="${VAULT_ADDR_INTERNAL:-http://vault:8200}"

# Ejecuta la CLI de Vault en un contenedor (no requiere instalarla en el host)
vault_cli() {
  : "${VAULT_TOKEN:?Define VAULT_TOKEN con un token de administrador}"
  docker run --rm -i --network "$VAULT_NETWORK" \
    -e VAULT_ADDR="$VAULT_ADDR_INTERNAL" -e VAULT_TOKEN \
    -v "$VAULT_DIR:/vault/agent:ro" \
    "$VAULT_IMAGE" vault "$@"
}
