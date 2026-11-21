# Deploying the API to Google Cloud Run

These are the steps I used to put the API online. Everything is in us-east1 and inside Google Cloud's free tier.

## Requirements

- A Google Cloud project with billing turned on (I used the Free Trial and a $1 budget alert)
- The gcloud CLI, logged in with `gcloud auth login`
- Docker Desktop
- A trained model in `models/` (run `python -m src.train` first)

## Steps

```bash
# 1. Point gcloud at the project and turn on the services
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com

# 2. Create a Docker repository in Artifact Registry and let Docker push to it
gcloud artifacts repositories create drone-clf --repository-format=docker --location=us-east1
gcloud auth configure-docker us-east1-docker.pkg.dev

# 3. Build the slim API image (it copies models/ in) and push it
docker build --target serve -t us-east1-docker.pkg.dev/PROJECT_ID/drone-clf/api:v1 .
docker push us-east1-docker.pkg.dev/PROJECT_ID/drone-clf/api:v1

# 4. Deploy it
gcloud run deploy drone-clf-api --image us-east1-docker.pkg.dev/PROJECT_ID/drone-clf/api:v1 \
  --region us-east1 --allow-unauthenticated --min-instances 0 --max-instances 1 \
  --memory 1Gi --cpu 1 --port 8080 --concurrency 10 --timeout 60
```

## Keeping it free

| Setting | Why |
|---|---|
| Region us-east1 | Cloud Storage and Cloud Run both have free tiers there |
| `--min-instances 0` | Nothing runs, or is billed, when nobody is using it. The first request after a quiet period takes a few seconds. |
| `--max-instances 1` | Caps the cost if someone floods the endpoint |
| Build locally, push the image | Skips Cloud Build |
| One image in Artifact Registry | The API image is 177 MB compressed, under the 0.5 GB free storage. Delete old versions after a redeploy. |

## Testing it

```bash
URL=$(gcloud run services describe drone-clf-api --region us-east1 --format "value(status.url)")
curl "$URL/health"
python scripts/make_request.py frontend/examples/two_wheeler_1.png --out request.json
curl -X POST "$URL/predict" -H "Content-Type: application/json" -d @request.json
```

## Shutting it down

```bash
gcloud run services delete drone-clf-api --region us-east1
gcloud artifacts repositories delete drone-clf --location us-east1
gcloud storage rm -r gs://drone-clf-data-gs
```