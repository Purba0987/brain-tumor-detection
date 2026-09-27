# 🚀 Server Deployment Guide: Render (Flask + Docker + TensorFlow)

This guide walks you through deploying your Flask backend server (`main.py`) with TensorFlow & PostgreSQL (Neon) to **Render** using Docker.

---

## 📋 Prerequisites
1. **GitHub Repository**: Ensure your latest code is pushed to GitHub:
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin main
   ```
2. **Render Account**: Create a free account at [dashboard.render.com](https://dashboard.render.com).

---

## 🛠️ Deployment Steps (Option A: Auto-deploy via render.yaml Blueprint)

We have created a [`render.yaml`](file:///c:/Users/Purba%20Hanra/brain%20tumor/render.yaml) blueprint configuration in your workspace!

1. Go to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top right corner and select **Blueprints**.
3. Connect your GitHub account and select your repository (`brain-tumor-detection`).
4. Render will automatically read [`render.yaml`](file:///c:/Users/Purba%20Hanra/brain%20tumor/render.yaml) and configure the Docker Web Service.
5. Fill in your environment variable secrets:
   - `DATABASE_URL`: Your Neon PostgreSQL connection string.
   - `GEMINI_API_KEY`: Your Gemini API Key.
   - `SMTP_EMAIL`: (Optional) Your Gmail address.
   - `SMTP_PASSWORD`: (Optional) Your Gmail App Password.
6. Click **Apply**. Render will build the Docker container and deploy it live!

---

## 🛠️ Deployment Steps (Option B: Manual Web Service Setup)

If you prefer configuring it manually through the Render dashboard UI:

1. Click **New +** -> **Web Service**.
2. Connect your repository (`Purba0987/brain-tumor-detection`).
3. Select **Docker** as the Runtime environment.
4. Set **Region** to your preferred location (e.g. `Singapore`).
5. Choose **Instance Type**: `Free` (or `Starter` for faster ML model loading).
6. Under **Environment Variables**, add:
   - `PORT`: `8000`
   - `SECRET_KEY`: Generate a secret key string
   - `DATABASE_URL`: `postgresql://...` (your Neon DB URL)
   - `GEMINI_API_KEY`: Your Gemini API Key
   - `SMTP_EMAIL`: Gmail address (for OTPs)
   - `SMTP_PASSWORD`: 16-character Gmail App Password
7. Click **Create Web Service**.

---

## 🔗 Connecting Deployed Frontend to Render Backend

Once Render finishes building, you will get a live backend URL (e.g., `https://brain-tumor-backend.onrender.com`).

1. Verify server health in your browser: `https://brain-tumor-backend.onrender.com/api/stats`
2. Update your frontend requests to point to `https://brain-tumor-backend.onrender.com` instead of `http://localhost:5000`.
3. Enjoy your live deployed application!
