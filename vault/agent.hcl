# Vault Agent: se autentica con AppRole, genera los archivos .env y termina.
# La dirección de Vault se toma de VAULT_ADDR (ver docker-compose.yml).

exit_after_auth = true

auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path                   = "/vault/approle/role_id"
      secret_id_file_path                 = "/vault/approle/secret_id"
      remove_secret_id_file_after_reading = false
    }
  }
}

template {
  source      = "/vault/agent/templates/house-price-predictor_prod.ctmpl"
  destination = "/workspace/.env"
  perms       = "0600"
}
