# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the backend and frontend together
FROM python:3.10-slim
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/dist ./dist

# Set working directory to backend so Flask static path "../dist" works
WORKDIR /app/backend

# Run the Flask app with Gunicorn, binding to the PORT environment variable provided by Railway
CMD ["sh", "-c", "gunicorn app:app -b 0.0.0.0:$PORT"]
