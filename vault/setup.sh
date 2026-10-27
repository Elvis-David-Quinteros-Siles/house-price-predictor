#!/usr/bin/env bash
# Crea el motor KV, la política y el AppRole del proyecto; guarda role_id/secret_id en vault/approle/
# Uso: VAULT_TOKEN=<token-admin> ./vault/setup.sh
source "$(dirname "$0")/lib.sh"

mounts="$(vault_cli secrets list -format=json)"
grep -q '"secret/"' <<<"$mounts" || vault_cli secrets enable -path=secret kv-v2
auths="$(vault_cli auth list -format=json)"
grep -q '"approle/"' <<<"$auths" || vault_cli auth enable approle

vault_cli policy write "$PROJECT" /vault/agent/policy.hcl
vault_cli write "auth/approle/role/$PROJECT" \
  token_policies="$PROJECT" token_ttl=15m token_max_ttl=1h secret_id_ttl=0

mkdir -p "$VAULT_DIR/approle" && chmod 700 "$VAULT_DIR/approle"
vault_cli read -field=role_id "auth/approle/role/$PROJECT/role-id" > "$VAULT_DIR/approle/role_id"
vault_cli write -f -field=secret_id "auth/approle/role/$PROJECT/secret-id" > "$VAULT_DIR/approle/secret_id"
chmod 600 "$VAULT_DIR/approle/"*
echo "AppRole '$PROJECT' listo. Credenciales en vault/approle/"
