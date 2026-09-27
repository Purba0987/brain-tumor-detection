<#
deploy_azure.ps1
Automates: resource group, ACR, service principal, App Service plan + Web App, sets GitHub Actions secrets, and pushes repo.

Prereqs (run locally):
- Azure CLI installed and logged in (`az login`)
- Git installed and authenticated to GitHub (or `gh` CLI installed and authenticated)
- GitHub CLI (`gh`) installed and authenticated (recommended)

Usage (PowerShell):
.\deploy_azure.ps1 -SubscriptionId <SUBSCRIPTION_ID> -ResourceGroup myResourceGroup -Location eastus -AcrName myacrname -AppName mywebappname -GitRemote https://github.com/Purba0987/brain-tumor-detection

You will be prompted before destructive operations.
#>

param(
    [Parameter(Mandatory=$true)] [string] $SubscriptionId,
    [Parameter(Mandatory=$false)] [string] $ResourceGroup = "myResourceGroup",
    [Parameter(Mandatory=$false)] [string] $Location = "eastus",
    [Parameter(Mandatory=$true)] [string] $AcrName,
    [Parameter(Mandatory=$true)] [string] $AppName,
    [Parameter(Mandatory=$false)] [string] $GitRemote = "https://github.com/Purba0987/brain-tumor-detection"
)

Write-Host "Setting Azure subscription to $SubscriptionId"
az account set --subscription $SubscriptionId

Write-Host "Creating resource group $ResourceGroup in $Location"
az group create --name $ResourceGroup --location $Location | Out-Null

Write-Host "Creating ACR ($AcrName)"
az acr create --resource-group $ResourceGroup --name $AcrName --sku Basic | Out-Null

Write-Host "Logging into ACR"
az acr login --name $AcrName

Write-Host "Creating service principal for GitHub Actions (sdk-auth JSON will be printed)"
$scope = "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup"
$spJson = az ad sp create-for-rbac --name "github-action-sp-$AcrName" --role contributor --scopes $scope --sdk-auth

if (-not $spJson) {
    Write-Error "Failed to create service principal. Exiting."
    exit 1
}

Write-Host "Service principal JSON created. Storing as GitHub Actions secret AZURE_CREDENTIALS..."

if (Get-Command gh -ErrorAction SilentlyContinue) {
    gh secret set AZURE_CREDENTIALS --body "$spJson"
    gh secret set ACR_LOGIN_SERVER --body "$($AcrName).azurecr.io"
    gh secret set WEBAPP_NAME --body "$AppName"
    Write-Host "GitHub secrets set via gh CLI."
} else {
    Write-Warning "gh CLI not found. Please create the following repository secrets manually in GitHub: AZURE_CREDENTIALS (paste JSON), ACR_LOGIN_SERVER ($($AcrName).azurecr.io), WEBAPP_NAME ($AppName)"
}

Write-Host "Creating App Service plan and Web App (Linux)"
az appservice plan create --name ${AppName}Plan --resource-group $ResourceGroup --is-linux --sku B1 | Out-Null

Write-Host "Creating Web App for Containers (container image will be updated by the workflow after push)"
az webapp create --resource-group $ResourceGroup --plan ${AppName}Plan --name $AppName --deployment-container-image-name "$($AcrName).azurecr.io/brain-tumor:latest" | Out-Null

Write-Host "Assign managed identity to Web App and grant AcrPull role"
az webapp identity assign --name $AppName --resource-group $ResourceGroup | Out-Null
$principalId = az webapp identity show --name $AppName --resource-group $ResourceGroup --query principalId -o tsv
$acrId = az acr show --name $AcrName --resource-group $ResourceGroup --query id -o tsv
az role assignment create --assignee $principalId --role AcrPull --scope $acrId | Out-Null

Write-Host "Setting App Setting PORT=8000"
az webapp config appsettings set --resource-group $ResourceGroup --name $AppName --settings PORT=8000 | Out-Null

Write-Host "Pushing repository to GitHub remote $GitRemote"
git add .
git commit -m "Prepare Azure container deployment" -a -q
git remote remove origin -ErrorAction SilentlyContinue
git remote add origin $GitRemote
git branch -M main
git push -u origin main

Write-Host "Done. The GitHub Actions workflow should trigger; monitor Actions tab. To see live logs from the webapp after deployment run:"
Write-Host "az webapp log tail --name $AppName --resource-group $ResourceGroup"

Write-Host "If the workflow builds and pushes successfully, your site URL will be: https://$AppName.azurewebsites.net"
