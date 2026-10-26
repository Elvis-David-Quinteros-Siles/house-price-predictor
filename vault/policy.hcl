# Política de solo lectura para house-price-predictor
path "secret/data/house-price-predictor/*" {
  capabilities = ["read"]
}

path "secret/metadata/house-price-predictor/*" {
  capabilities = ["read", "list"]
}
