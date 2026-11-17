# CLOUD DEPLOYMENT GUIDE
## Cricket Pressure Performance Intelligence Dashboard

This guide explains how to deploy the interactive dashboard online so anyone can access it via a public HTTPS URL.

All required deployment configuration files (`Procfile`, `Dockerfile`, `render.yaml`, `railway.json`, `requirements.txt`) have already been generated inside this folder.

---

### Option 1: Render.com (Recommended — 100% Free & Easiest)

Render offers free hosting for Python web apps with automatic HTTPS certificates.

1. **Push Dashboard folder to a GitHub Repository:**
   ```bash
   cd "Pressure_Performance_Intelligence/Supporting_Files/Dashboard"
   git init
   git add .
   git commit -m "Initial commit for Cricket Pressure Intelligence Dashboard"
   git branch -M main
   # Create a new repository on GitHub (e.g. cricket-pressure-dashboard) and push:
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. Go to **[Render.com](https://render.com/)** and sign in with GitHub.
3. Click **"New +"** → **"Web Service"**.
4. Connect your GitHub repository.
5. Configure the service settings (or let Render read `render.yaml` automatically):
   - **Name:** `cricket-pressure-dashboard`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server --workers 2 --threads 2 --timeout 120 -b 0.0.0.0:$PORT`
   - **Plan:** Free ($0/month)
6. Click **"Create Web Service"**.
7. In ~2 minutes, your dashboard will be live at:  
   `https://cricket-pressure-dashboard.onrender.com`

---

### Option 2: Railway.app (One-Click Git / CLI Deploy)

1. Sign in to **[Railway.app](https://railway.app/)**.
2. Click **"New Project"** → **"Deploy from GitHub repo"**.
3. Select your repository.
4. Railway will automatically detect `railway.json` and `Procfile`, install dependencies, and launch `gunicorn`.
5. Under service settings, click **"Generate Domain"** to get a public URL like:  
   `https://cricket-pressure-production.up.railway.app`

---

### Option 3: Hugging Face Spaces (Free Cloud Hosting with 16GB RAM)

1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)**.
2. Click **"Create new Space"**.
3. Choose:
   - **Space SDK:** `Docker` (Blank)
   - **Hardware:** CPU basic (Free - 2 vCPU, 16GB RAM)
4. Push the contents of `Supporting_Files/Dashboard/` to the space's git repository.
5. Hugging Face builds the included `Dockerfile` and deploys your dashboard automatically at:  
   `https://huggingface.co/spaces/<username>/<space-name>`

---

### Option 4: Instant Public Live Link Right Now (Without Cloud Accounts)

If you want an immediate public HTTPS URL to test or share right this moment from your machine:

#### Using Cloudflare Tunnel (`cloudflared`):
```bash
# 1. Start your local dashboard
python app.py

# 2. In another terminal, run cloudflared quick tunnel:
npx cloudflared tunnel --url http://127.0.0.1:8050
```
This gives you an instant free temporary public HTTPS link (e.g. `https://xyz.trycloudflare.com`).

#### Using Localtunnel:
```bash
# In another terminal:
npx localtunnel --port 8050
```
This provides an instant public URL (e.g. `https://cricket-scouting.loca.lt`).
