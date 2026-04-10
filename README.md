# 🚀 MLOps: CI/CD + FastAPI + Amazon S3

Proyecto de MLOps que implementa un flujo completo desde entrenamiento hasta despliegue automatizado de un modelo de Machine Learning.

---

## 📁 Estructura del Proyecto

```
mi-proyecto-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Pipeline CI (lint + tests)  Susana
│       ├── cd.yml          # Pipeline CD (deploy a EC2)  Susana
│       └── retrain.yml     # Reentrenamiento             Kendra
├── app/
│   ├── main.py             # API FastAPI                 Kendra
│   └── utils.py            # Funciones auxiliares        Kendra
├── model/
│   └── .gitkeep            # El modelo se descarga desde S3 / Kendra
├── scripts/
│   ├── train.py            # Script de entrenamiento      Kendra
│   └── setup_ec2.sh        # Setup inicial de la instancia EC2  / Susana
├── tests/
│   └── test_api.py         # Tests de la API
├── Dockerfile                                            Susana
├── requirements.txt                                      Susana
├── pyproject.toml                                        Kendra
└── README.md                                             Primera parte Susana.
```

---

## ⚙️ Configuración Inicial

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/mi-proyecto-mlops.git
cd mi-proyecto-mlops
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 3. Variables de entorno (local)

Crear un archivo `.env` (nunca subir a git):

```env
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=nombre-de-tu-bucket
```

---

## 🔐 GitHub Secrets Necesarios

Ir a `Settings → Secrets and variables → Actions` y agregar:

| Secret | Descripción |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS |
| `AWS_REGION` | Región AWS (ej: `us-east-1`) |
| `AWS_ACCOUNT_ID` | ID de la cuenta AWS (12 dígitos) |
| `ECR_REPOSITORY` | Nombre del repositorio en ECR |
| `S3_BUCKET_NAME` | Nombre del bucket S3 |
| `EC2_HOST` | IP pública de la instancia EC2 |
| `EC2_USER` | Usuario SSH (normalmente `ubuntu`) |
| `EC2_SSH_KEY` | Contenido completo del archivo `.pem` |

---

## 🖥️ Configurar EC2

### Crear la instancia (AWS Console)

1. Ir a **EC2 → Launch Instance**
2. Elegir **Ubuntu Server 22.04 LTS**
3. Tipo: `t2.micro` (free tier) o `t3.small` recomendado
4. **Security Group** — abrir puertos:
   - `22` (SSH) — solo desde tu IP
   - `8000` (API) — desde cualquier lugar (`0.0.0.0/0`)
5. Crear o usar un **Key Pair** existente (guardar el `.pem`)
6. Lanzar la instancia

### Configurar la instancia

```bash
# Conectarse por SSH
chmod 400 tu-key.pem
ssh -i tu-key.pem ubuntu@IP_PUBLICA_EC2

# Ejecutar el script de setup
bash scripts/setup_ec2.sh
```

---

## 🔄 Flujo de CI/CD

```
Push a cualquier rama
        │
        ▼
   ┌─────────┐
   │  CI.yml │  ← flake8 lint + pytest
   └────┬────┘
        │ ✅ pasa
        ▼
   Pull Request → main
        │
        ▼
   ┌─────────┐
   │  CD.yml │  ← build Docker → push ECR → deploy EC2
   └─────────┘
```

### Pipeline CI (`ci.yml`)
- Se ejecuta en **todo push** y en **PRs hacia `main`/`develop`**
- Corre `flake8` para lint
- Corre `pytest` para tests

### Pipeline CD (`cd.yml`)
- Se ejecuta solo al hacer **merge/push a `main`**
- Build de imagen Docker
- Push a Amazon ECR
- Deploy automático en EC2 vía SSH

---

## 🌿 Branching Strategy

```
main          ← producción, protegida
  └── develop ← integración
        ├── feature/nombre-feature
        ├── fix/nombre-fix
        └── chore/nombre-tarea
```

**Flujo de trabajo:**
1. Crear rama desde `develop`: `git checkout -b feature/mi-feature`
2. Hacer commits pequeños y descriptivos
3. Abrir Pull Request hacia `develop`
4. Revisión de código (code review)
5. Merge a `develop` → luego PR de `develop` a `main` para producción

---

## 🐳 Correr localmente con Docker

```bash
# Build
docker build -t mlops-app .

# Run
docker run -d \
  --name mlops-app \
  -p 8000:8000 \
  --env-file .env \
  mlops-app

# Ver logs
docker logs -f mlops-app
```

La API estará disponible en: `http://localhost:8000`  
Documentación automática: `http://localhost:8000/docs`

---

## 🧪 Correr Tests

```bash
pytest tests/ -v
```
