# ─────────────────────────────────────────────────────────────────
# 🐳 Nesine Futbol Tahmin Sistemi — Production Dockerfile
# Python 3.11 + Google Chrome (headless) + Chromedriver
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim-bookworm AS base

# Ortam değişkenleri
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CHROME_BIN=/usr/bin/google-chrome-stable \
    DISPLAY=:99

# ─── Sistem bağımlılıkları + Google Chrome ───────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    curl \
    unzip \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    xdg-utils \
    libxss1 \
    libgconf-2-4 \
    libappindicator3-1 \
    && rm -rf /var/lib/apt/lists/*

# Google Chrome Stable kurulumu
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Chromedriver kurulumu (Chrome versiyonuyla uyumlu)
RUN CHROME_VERSION=$(google-chrome-stable --version | grep -oP '\d+\.\d+\.\d+') \
    && DRIVER_URL="https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_${CHROME_VERSION%.*}" \
    && DRIVER_VERSION=$(curl -sS "$DRIVER_URL" 2>/dev/null || echo "") \
    && if [ -z "$DRIVER_VERSION" ]; then \
         MAJOR=$(echo "$CHROME_VERSION" | cut -d. -f1); \
         DRIVER_VERSION=$(curl -sS "https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_${MAJOR}" 2>/dev/null || echo ""); \
       fi \
    && if [ -n "$DRIVER_VERSION" ]; then \
         wget -q "https://storage.googleapis.com/chrome-for-testing-public/${DRIVER_VERSION}/linux64/chromedriver-linux64.zip" -O /tmp/chromedriver.zip \
         && unzip -q /tmp/chromedriver.zip -d /tmp/ \
         && mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver \
         && chmod +x /usr/local/bin/chromedriver \
         && rm -rf /tmp/chromedriver*; \
       else \
         echo "WARN: chromedriver not auto-installed, will use webdriver-manager at runtime"; \
       fi

# ─── Çalışma dizini ──────────────────────────────────────────────
WORKDIR /app

# ─── Python bağımlılıkları ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Uygulama kodunu kopyala ─────────────────────────────────────
COPY . .

# Gerekli dizinleri oluştur
RUN mkdir -p models_cache logs

# ─── Sağlık kontrolü ─────────────────────────────────────────────
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "from database import engine; engine.connect()" || exit 1

# ─── Varsayılan komut ────────────────────────────────────────────
CMD ["python", "main.py"]
