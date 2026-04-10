#!/bin/bash
# =============================================================================
# setup_ec2.sh — Configura una instancia EC2 Ubuntu 22.04 desde cero
# Ejecutar como: bash setup_ec2.sh
# =============================================================================

set -e  # Detener si hay cualquier error

echo "=============================="
echo " Setup MLOps EC2 - Ubuntu 22.04"
echo "=============================="

# 1. Actualizar paquetes del sistema
echo "[1/5] Actualizando paquetes..."
sudo apt-get update -y && sudo apt-get upgrade -y

# 2. Instalar Docker
echo "[2/5] Instalando Docker..."
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Agregar usuario ubuntu al grupo docker (sin necesitar sudo)
sudo usermod -aG docker ubuntu
sudo systemctl enable docker
sudo systemctl start docker

# 3. Instalar AWS CLI v2
echo "[3/5] Instalando AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt-get install -y unzip
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws/

# 4. Verificar instalaciones
echo "[4/5] Verificando instalaciones..."
docker --version
aws --version

# 5. Mensaje final
echo "[5/5] ¡Setup completado!"
echo ""
echo "IMPORTANTE: Cierra y vuelve a abrir la sesión SSH para"
echo "que los permisos de Docker se apliquen correctamente."
echo ""
echo "Luego configura las credenciales AWS con:"
echo "  aws configure"
echo "O mejor aún, asigna un IAM Role a esta instancia EC2."
