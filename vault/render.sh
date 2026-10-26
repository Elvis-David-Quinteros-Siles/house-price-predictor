#!/usr/bin/env bash
# Genera los .env del proyecto desde Vault usando Vault Agent + AppRole
# Uso: ./vault/render.sh && docker compose up -d
source "$(dirname "$0")/lib.sh"
[ -f "$VAULT_DIR/approle/role_id" ] || { echo "Falta vault/approle/; ejecuta antes ./vault/setup.sh" >&2; exit 1; }
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose -f "$VAULT_DIR/docker-compose.yml" run --rm vault-agent
echo "Archivos .env generados desde Vault:"
cut -d' ' -f1 "$VAULT_DIR/targets" | sed 's/^/  - /'
