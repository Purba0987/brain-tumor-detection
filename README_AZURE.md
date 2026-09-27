# Deploying `brain-tumor-detection` to Azure App Service (Container)

This guide shows how to build the Docker image, push it to Azure Container Registry (ACR), and deploy to Azure App Service for Containers.

Prerequisites
- Azure CLI installed and logged in: `az login`
- Docker installed locally
- An Azure subscription

Steps

1) Create a resource group

```bash
az group create --name myResourceGroup --location eastus
```

2) Create an Azure Container Registry (ACR)

```bash
az acr create --resource-group myResourceGroup --name <ACR_NAME> --sku Basic
az acr login --name <ACR_NAME>
```

3) Build and push the Docker image

```bash
docker build -t <ACR_NAME>.azurecr.io/brain-tumor:latest .
docker push <ACR_NAME>.azurecr.io/brain-tumor:latest
```

4) Create an App Service plan (Linux) and a Web App for Containers

```bash
az appservice plan create --name myPlan --resource-group myResourceGroup --is-linux --sku B1
az webapp create --resource-group myResourceGroup --plan myPlan --name <APP_NAME> --deployment-container-image-name <ACR_NAME>.azurecr.io/brain-tumor:latest
```

5) Configure the web app to pull from ACR (if needed), and assign ACR pull permissions

```bash
# Configure container (if you want to explicitly set it)
az webapp config container set --name <APP_NAME> --resource-group myResourceGroup --docker-custom-image-name <ACR_NAME>.azurecr.io/brain-tumor:latest --docker-registry-server-url https://<ACR_NAME>.azurecr.io

# If webapp cannot pull from ACR, give it access (managed identity)
az webapp identity assign --name <APP_NAME> --resource-group myResourceGroup
PRINCIPAL_ID=$(az webapp identity show --name <APP_NAME> --resource-group myResourceGroup --query principalId -o tsv)
acrId=$(az acr show --name <ACR_NAME> --resource-group myResourceGroup --query id -o tsv)
az role assignment create --assignee $PRINCIPAL_ID --role AcrPull --scope $acrId
```

6) Set environment variables (if needed)

```bash
az webapp config appsettings set --resource-group myResourceGroup --name <APP_NAME> --settings PORT=8000
```

7) View logs / stream

```bash
az webapp log tail --name <APP_NAME> --resource-group myResourceGroup
```

Notes
- The app listens on the `PORT` environment variable; the Docker image runs Gunicorn bound to `${PORT:-8000}`.
- The repo includes an `ffmpeg` folder; the Docker image installs system `ffmpeg` which should work for the `/speech-to-text` route.
- `requirements.txt` includes `tensorflow` which will increase image size — consider using a GPU-enabled image or an external model service for production.

If you want, I can:
- Build and test the Docker image locally and run it.
- Create a GitHub Actions workflow to automate build & push to ACR.
- Prepare an `az` script that creates resources and deploys automatically.
