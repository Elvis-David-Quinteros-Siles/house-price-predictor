#!/usr/bin/env bash
# Funciones comunes para los scripts de Vault de house-price-predictor
set -euo pipefail
VAULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$VAULT_DIR")"
PROJECT="house-price-predictor"
# vault/local.env (no versionado) permite ajustar variables por servidor, p. ej. VAULT_RENDER_ROOT
[ -f "$VAULT_DIR/local.env" ] && source "$VAULT_DIR/local.env"
# Carpeta donde viven los .env del despliegue (por defecto, la raíz del repo)
RENDER_ROOT="${VAULT_RENDER_ROOT:-$ROOT_DIR}"
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
