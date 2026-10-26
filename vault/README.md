# HashiCorp Vault — house-price-predictor

Los secretos de este proyecto se guardan en Vault (`https://vault.edqs.online`, motor KV v2 `secret/`)
y se inyectan como archivos `.env` mediante **Vault Agent** autenticado con **AppRole**.
Los `.env` dejan de ser la fuente de verdad: se generan desde Vault y nunca se versionan.

| Archivo .env | Ruta en Vault |
|---|---|
| `.env` | `secret/house-price-predictor/prod` |

## Puesta en marcha (una vez)

```bash
export VAULT_TOKEN=<token-de-administrador>
./vault/setup.sh   # motor KV, política 'house-price-predictor' (solo lectura) y AppRole
./vault/seed.sh    # sube los .env actuales a Vault
unset VAULT_TOKEN
```

`setup.sh` deja `role_id` y `secret_id` en `vault/approle/` (ignorado por git, permisos 600).

## Despliegue

```bash
./vault/render.sh        # Vault Agent genera los .env desde Vault
docker compose up -d
```

## Rotar un secreto

```bash
VAULT_TOKEN=<token> docker run --rm --network proxy -e VAULT_ADDR=http://vault:8200 -e VAULT_TOKEN \
  hashicorp/vault:2.1.1 vault kv patch secret/<ruta> CLAVE=nuevo-valor
./vault/render.sh && docker compose up -d
```

## Archivos

- `policy.hcl`: política de mínimo privilegio (lectura de `secret/house-price-predictor/*`).
- `agent.hcl`, `templates/`: configuración de Vault Agent y plantillas de los `.env`.
- `docker-compose.yml`: contenedor efímero de Vault Agent en la red `proxy`.
- `targets`: correspondencia archivo `.env` → ruta en Vault.
