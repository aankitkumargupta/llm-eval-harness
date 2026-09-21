# The browser UI (python main.py serve): landing page, sign-in, product pages
# and every screen, served by the stdlib server. One process, no framework.
#
# Build:  docker build -t llm-eval-harness .
# Run:    see DEPLOYMENT.md, "The web UI in a container". Never bake keys or
#         the pilot password into the image; pass them at run time.
FROM python:3.12-slim

# pdfplumber (dataset preparation) needs these; drop them if you never use
# prepare_dataset.py inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8010

# Bind to every interface inside the container; publish the port to loopback
# on the host and put a reverse proxy with TLS in front (DEPLOYMENT.md).
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8010", "--no-browser"]
