#!/bin/bash

# Quick deployment script for Google Cloud Run
set -e

echo "Starting deployment to Google Cloud Run..."

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "No project set. Please run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "Using project: $PROJECT_ID"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy using Cloud Build
echo "Starting Cloud Build deployment..."
gcloud builds submit --config cloudbuild.yaml .

# Get the service URL
echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe churn-prediction-api \
    --region=us-central1 \
    --format="value(status.url)")

echo "Deployment completed!"
echo "Service URL: $SERVICE_URL"
echo "Health check: $SERVICE_URL/health"
echo "API docs: $SERVICE_URL/docs"

# Test the deployment
echo "Testing deployment..."
curl -s "$SERVICE_URL/health" && echo "Health check passed!" || echo "Health check failed!"
