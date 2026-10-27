#!/usr/bin/env bash
# Sube los .env actuales del proyecto a Vault (secret/<proyecto>/...)
# Uso: VAULT_TOKEN=<token-admin> ./vault/seed.sh
source "$(dirname "$0")/lib.sh"

while read -r envfile kvpath; do
  file="$RENDER_ROOT/$envfile"
  if [ ! -f "$file" ]; then echo "Omitido: $envfile no existe"; continue; fi
  python3 - "$file" <<'PY' | vault_cli kv put "secret/$kvpath" -
import json, re, sys
data = {}
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, val = line.split("=", 1)
    key = re.sub(r"^export\s+", "", key).strip()
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        val = val[1:-1]
    data[key] = val
json.dump(data, sys.stdout)
PY
  echo "Cargado: $envfile -> secret/$kvpath"
done < "$VAULT_DIR/targets"
