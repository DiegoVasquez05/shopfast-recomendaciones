#!/usr/bin/env bash
# Prepara una EC2 (Amazon Linux 2023, t3.large recomendado: 2 vCPU / 8 GB) para el piloto de TelecomBot AI.
# Uso dentro de la EC2:
#   curl -fsSL https://raw.githubusercontent.com/DiegoVasquez05/shopfast-recomendaciones/main/infra/ec2-setup.sh | bash
set -euo pipefail
REPO="${REPO:-https://github.com/DiegoVasquez05/shopfast-recomendaciones.git}"
DIR="${DIR:-$HOME/telecombot}"

sudo dnf update -y
sudo dnf install -y docker git openssl make python3.11 python3.11-pip
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -fsSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

[ -d "$DIR" ] || git clone "$REPO" "$DIR"
cd "$DIR"
[ -f .env ] || cp .env.example .env
python3.11 -m venv .venv && .venv/bin/pip install -q -r requirements-dev.txt
PUBLIC_IP=$(curl -fs http://checkip.amazonaws.com || echo localhost)
./nginx/gen-cert.sh "$PUBLIC_IP"
cat <<MSG

Listo. Próximos pasos (ver docs/despliegue-aws.md):
  1) Editar .env (secretos) y copiar el dataset Bitext a data/bitext.csv
  2) source .venv/bin/activate && make train BITEXT=data/bitext.csv
  3) newgrp docker  &&  docker compose up -d --build
  4) Abrir https://$PUBLIC_IP/widget , https://$PUBLIC_IP/docs y https://$PUBLIC_IP/dashboard/
MSG
