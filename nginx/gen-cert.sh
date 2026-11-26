#!/usr/bin/env bash
# Genera un certificado autofirmado para TLS (demo). En producción usar ACM o Let's Encrypt.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p certs
CN="${1:-localhost}"
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
  -keyout certs/server.key -out certs/server.crt \
  -subj "/C=CL/O=TelcoAustral/CN=${CN}" -addext "subjectAltName=DNS:${CN},DNS:localhost,IP:127.0.0.1"
echo "Certificado generado en nginx/certs (CN=${CN})"
