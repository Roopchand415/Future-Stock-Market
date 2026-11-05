# Railway Single-Service Deployment Guide

Great news! The codebase has been optimized to deploy as a **single, unified Docker container**. 
This makes deployment 100% foolproof, completely eliminates CORS and URL routing errors, avoids Nixpacks detection bugs, and builds extremely fast because we optimized the Python requirements.

## Deployment Steps

1. Go to your [Railway Dashboard](https://railway.app/dashboard).
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Choose your `Future-Stock-market-Prediction` repository.
4. **That's it!** Railway will automatically detect the `Dockerfile` at the root and build both the frontend and backend perfectly.
5. Once the build finishes, go to the **Networking** tab of your service and click **Generate Domain**.

## Why this is extremely reliable:
- **No Nixpacks Guesswork:** Railway will build the exact steps specified in the `Dockerfile`.
- **Fast Build Times:** We removed `tensorflow` from `requirements.txt` to prevent RAM crashes and build hangs on Railway. The backend will automatically fall back to a lightweight, fast Scikit-Learn neural network.
- **Unified Port:** Flask serves the built React frontend on the exact same port provided by Railway.

