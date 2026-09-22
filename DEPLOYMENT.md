# Deployment guide

How to run this harness for real, on a laptop, on a shared box, in a container,
and in CI. Includes the things that bite people: exposing the app, the embedded
vector store's single-writer rule, and how to stop a mis-typed config from
spending money all night.

**Read [Security](#security) before putting this on any machine other people can
reach.** The app accepts an API key and spends money; its sign-in is a shared
pilot password (set `HARNESS_PILOT_PASSWORD`), not real authentication, by design.

---

## Contents

- [Pick a deployment mode](#pick-a-deployment-mode)
- [Mode A: Local, single user](#mode-a--local-single-user)
- [Mode B: Shared server for a team](#mode-b--shared-server-for-a-team)
- [Mode C: Docker](#mode-c--docker)
- [Mode D: CI regression gate](#mode-d--ci-regression-gate)
- [Mode E: Scheduled runs](#mode-e--scheduled-runs)
- [Mode F: Quick demo link from a laptop](#mode-f-quick-demo-link-from-a-laptop)
- [Security](#security)
- [Cost controls](#cost-controls)
- [Data, state and backup](#data-state-and-backup)
- [Upgrading an existing install](#upgrading-an-existing-install)
- [Operational limits](#operational-limits)
- [Health checks](#health-checks)
- [Production checklist](#production-checklist)

---

## Pick a deployment mode

| Mode | Who it's for | Vector store | Auth needed |
|---|---|---|---|
| **A. Local** | One person evaluating models | Embedded (a folder) | The built-in sign-in |
| **B. Shared server** | A team reading results | Qdrant server | **Yes**: password set, TLS proxy |
| **C. Docker** | Reproducible / cloud | Either | **Yes**, if exposed |
| **D. CI gate** | Blocking regressions on PRs | Server or none | n/a |
| **E. Scheduled** | Nightly drift tracking | Server | n/a |
| **F. Quick demo link** | Showing stakeholders from a laptop | Embedded | Password set, `--budget` set |

Most people want **A** for running evaluations and **D** for protecting the
main branch. B, C and F matter when the site needs to be reached by others.

---

## Mode A: Local, single user

The default, and what the app is designed for. No Docker, no server, no ports
beyond Streamlit's own.

```bash
git clone <your-repo-url>
cd llm-eval-harness
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` in the project root (it is gitignored):

```
TOGETHER_API_KEY=your_together_key
OPENROUTER_API_KEY=your_openrouter_key
```

Verify before spending anything, the whole suite runs offline against a fake
provider, so it needs no key and costs nothing:

```bash
pytest
```

Then check the config and the provider routing:

```bash
python main.py validate --profile regulated_qa
```

Add the sign-in password to the same `.env`:

```
HARNESS_PILOT_PASSWORD=choose_something_private
```

Launch the web UI:

```bash
python main.py serve --port 8010 --budget 5
```

It opens at `http://127.0.0.1:8010`, on the landing page; **Sign in** takes a name and
function (attribution only, written to run manifests), a role and the password. Everything is
written under `workspace/` and `runs/`. Qdrant runs **embedded**, a folder, not a server.
`--budget` is the hard USD ceiling for runs started from the UI; leave it off only on a
machine nobody else can reach.

### What the UI does, in order

The **task bar** holds the working screens; the **sidebar** groups the site as Product
(what the platform is and its honest status), Evaluations (case studies, the Evaluate
wizard, saved reports) and Workspace (every screen again). A new user follows the numbers:

1. **Evaluate**, pick the kind of task, read the data format with sample rows, have the
   profile written and validated, upload your own JSONL. Free.
2. **Preflight**, `validate` and `estimate` for the profile: every provider, every price,
   the judge calls included. Free.
3. **Retrieval** and **Probes** (RAG profiles), build the index and generate adversarial
   items. Indexing embeds every chunk, so it costs money once per dataset.
4. **Run**, pick models, start the paired matrix in the background (Assurance Lead only).
5. **Analyse**, results with intervals, the decision, the gate, the arena; **Saved reports**
   freeze a run with its caveats; the HTML export is self-contained.

The older Streamlit app (`streamlit run app.py`, port 8501) is still shipped and has no
sign-in of its own.

---

## Mode B: Shared server for a team

Two things change: Qdrant becomes a real server, and **you must put
authentication in front of the app**.

### 1. Run Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Point the harness at it in `configs/run.yaml`:

```yaml
qdrant_url: "http://localhost:6333"
# leave qdrant_path unset/commented so the server is used
```

### 2. Split the roles

The web UI does this itself. Everyone signs in with the shared password; the role they pick
is enforced by the server: an **Evaluator** reads everything and cannot start a paid run, an
**Assurance Lead** can, up to `--budget`. Run one server, bound to loopback:

```bash
HARNESS_PILOT_PASSWORD=... python main.py serve --host 127.0.0.1 --port 8010 --budget 20 --no-browser
```

The older read-only Streamlit dashboard is still available for a team that only wants
numbers (`streamlit run dashboard.py --server.port 8502`); it makes no API calls.

### 3. Put a reverse proxy in front

The sign-in is a pilot gate, not accounts, so terminate TLS at the proxy and add its own
authentication if the audience is wider than the people who hold the password:

```nginx
server {
    listen 443 ssl;
    server_name eval.internal.example.com;

    ssl_certificate     /etc/ssl/certs/eval.crt;
    ssl_certificate_key /etc/ssl/private/eval.key;

    auth_basic           "Evaluation Intelligence";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8010;   # the web UI
        proxy_http_version 1.1;
        proxy_set_header Host       $host;
        proxy_read_timeout 600;             # job polling is plain HTTP; no websockets needed
    }
}
```

(For the Streamlit dashboard instead, proxy to 8502 and add the `Upgrade` / `Connection
"upgrade"` headers, since Streamlit needs websockets.) Keep the server bound to `127.0.0.1`
so it is unreachable except through the proxy.

---

## Mode C: Docker

No Dockerfile ships with the repo, because the right base image depends on
whether you want the dataset-prep extras. This one covers everything:

```dockerfile
FROM python:3.12-slim

# pdfplumber needs these; drop them if you never use prepare_dataset.py
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Never bake keys into an image. Pass them at run time.
ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# Bind to 0.0.0.0 so the container is reachable, put auth in front of it.
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true"]
```

```bash
docker build -t llm-eval-harness .

docker run -d --name eval \
  -p 127.0.0.1:8501:8501 \
  -e TOGETHER_API_KEY="$TOGETHER_API_KEY" \
  -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  -v "$PWD/workspace:/app/workspace" \
  llm-eval-harness
```

Two details that matter:

- **`-p 127.0.0.1:8501:8501`**, not `-p 8501:8501`. The second form publishes to
  every interface, and on a cloud host that means the public internet.
- **Mount `workspace/`.** Everything the harness produces lives there. Without
  the volume, your index, cache and results die with the container.

With Qdrant as a separate service:

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant
    volumes: ["qdrant_storage:/qdrant/storage"]

  harness:
    build: .
    ports: ["127.0.0.1:8501:8501"]
    environment:
      TOGETHER_API_KEY: ${TOGETHER_API_KEY}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY}
    volumes: ["./workspace:/app/workspace"]
    depends_on: [qdrant]

volumes:
  qdrant_storage:
```

Then set `qdrant_url: "http://qdrant:6333"` in `configs/run.yaml`.

---

### The web UI in a container

The Dockerfile at the repo root serves the browser UI (`python main.py serve`: landing
page, sign-in, product pages and every screen) on port 8010. Keys and the pilot password are
passed at run time, never baked in; `.dockerignore` keeps `.env`, `workspace/` and `runs/` out
of the image.

```bash
docker build -t llm-eval-harness .

docker run -d --name eval-web   -p 127.0.0.1:8010:8010   -e TOGETHER_API_KEY="$TOGETHER_API_KEY"   -e HARNESS_PILOT_PASSWORD="$HARNESS_PILOT_PASSWORD"   -v "$PWD/workspace:/app/workspace"   -v "$PWD/runs:/app/runs"   llm-eval-harness
```

Then put a reverse proxy with TLS in front of `127.0.0.1:8010` (Caddy, nginx, or the cloud
host's ingress). The sign-in gate is a shared pilot password with server-side roles, not
accounts: set `HARNESS_PILOT_PASSWORD` to something private before the port is reachable
by anyone else, and add `--budget <usd>` to the `CMD` if runs may be started from the UI.

## Mode D: CI regression gate

This is the deployment most teams get the most out of. It answers "did my prompt
change break anything?" on every PR.

The gate compares a candidate run against a stored baseline and **exits
non-zero** on regression. It only fires when a drop is both beyond tolerance
*and* statistically significant, so it does not cry wolf on small evalsets.

```yaml
# .github/workflows/eval-gate.yml
name: LLM eval gate

on: pull_request

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install -r requirements.txt

      # The trace store is the baseline. Cache or restore it from wherever you
      # keep it, an artifact, S3, or committed for a small evalset.
      - name: Restore baseline traces
        uses: actions/cache@v4
        with:
          path: runs/traces
          key: eval-traces-${{ github.base_ref }}

      - name: Run the evaluation
        env:
          TOGETHER_API_KEY: ${{ secrets.TOGETHER_API_KEY }}
        run: |
          python main.py run --profile regulated_qa \
            --run-id "pr-${{ github.event.pull_request.number }}" \
            --budget 5.00

      - name: Gate on regressions
        run: |
          python main.py gate --profile regulated_qa \
            --baseline "$KNOWN_GOOD_RUN_ID" \
            --candidate "pr-${{ github.event.pull_request.number }}" \
            --gate accuracy:0.02 \
            --gate faithfulness:0.01 \
            --floor "injection_resisted>=0.95" \
            --max-cost 0.005
```

Notes from experience:

- **Always pass `--budget`.** A runaway CI job with an API key is the most
  expensive failure mode this tool has.
- **`--strict`** turns off the significance requirement and fails on any raw
  drop. Use it only if your evalset is large enough that noise is not a factor.
- The whole *unit* suite (`pytest`) needs no key and no network, run it on
  every push; run the eval gate only where it earns its cost.

---

## Mode E: Scheduled runs

Hosted models change under you. A nightly run turns that from a surprise into a
graph.

```cron
# 03:00 daily; append to the same store so runs are comparable over time
0 3 * * * cd /opt/llm-eval-harness && \
  /opt/llm-eval-harness/.venv/bin/python main.py run \
    --profile regulated_qa --budget 10.00 \
    >> /var/log/llm-eval/run.log 2>&1
```

`python main.py runs` lists every recorded run; the dashboard's filters read the
same store, so drift shows up without extra tooling.

---

## Mode F: Quick demo link from a laptop

For showing the site to stakeholders for an afternoon without a host. A Cloudflare quick
tunnel needs no account and gives a public `trycloudflare.com` address that works while the
laptop, the server and the tunnel are all running.

```bash
winget install Cloudflare.cloudflared          # once; brew install cloudflared on macOS

python main.py serve --port 8010 --budget 5 --no-browser
cloudflared tunnel --url http://127.0.0.1:8010
```

The public address is printed by the second command. Before starting the tunnel:

- set `HARNESS_PILOT_PASSWORD` in `.env`; the built-in default is public;
- start the server with `--budget`; anyone with the link and the password who picks the
  Assurance Lead role can start a paid run up to that ceiling;
- share the password by a channel other than the link.

The address changes every time the tunnel restarts and disappears when it stops, so this is a
demo, not a deployment; use Mode B or C for anything that should stay up.

---

## Security

**The app is not hardened for exposure, and is not meant to be.** Treat it as a
desktop tool that happens to render in a browser.

| Risk | Why it exists | What to do |
|---|---|---|
| **A shared password, not accounts** | The sign-in is a pilot gate: one password, roles chosen at sign-in, sessions in process memory | Set `HARNESS_PILOT_PASSWORD`; bind to `127.0.0.1`; TLS and, for a wider audience, the proxy's own auth |
| **Spends money** | An Assurance Lead session can start a paid run | Start with `--budget`; keep the Assurance Lead role to the people who own the spend |
| **Accepts an API key** | The sidebar takes a key for convenience | Prefer the environment; the field is per-session and never written to disk |
| **Reads uploaded files** | PDFs are parsed and indexed | Only ingest documents you trust to that extent |
| **Corpus is an attack surface** | Retrieved text reaches the model as tokens | This is what the injection probes measure, run them |

Other rules worth keeping:

- **Never bake keys into a Docker image or commit them.** `.env` is gitignored;
  keep it that way.
- **`workspace/` contains your corpus, your answers and your prompts.** It is
  gitignored because it is often sensitive, not just because it is large.
- **Rotate the key** used for a shared deployment separately from a personal one,
  so you can revoke it without disrupting your own work.
- The harness makes outbound calls only to the providers you configure.

---

## Cost controls

Four layers, in the order they take effect:

1. **`estimate` before running.** Includes judge calls, which are usually the
   larger half of a judge-scored profile's bill.
   ```bash
   python main.py estimate --profile regulated_qa
   ```
2. **The budget ceiling.** `--budget 25.00`, or `budget_usd` in
   `configs/run.yaml`, or the sidebar field. The run aborts cleanly the moment
   measured spend crosses it, keeps every row already written, and stops issuing
   calls immediately rather than paying for a matrix it will discard.
3. **Caching.** Generations, judge verdicts and query embeddings are all cached
   by content hash, so a re-run of an unchanged config costs nothing. Keep
   `workspace/cache/` (or `.cache/`) between runs.
4. **`--max-cost` in the gate.** Catches the regression nobody reviews for: a
   prompt that grew 400 tokens looks fine in a diff and doubles your bill.

The adapted (tuning) pass is roughly `tuning_budget` times the size of a
baseline pass. It is off by default in `configs/run.yaml` for that reason.

---

## Data, state and backup

Everything the harness produces lives under `workspace/` (app) or the paths in
`configs/run.yaml` (CLI):

| Path | What it is | Back up? | Safe to delete? |
|---|---|---|---|
| `workspace/traces/` | Results, one Parquet part per checkpoint | **Yes** | No, this is the product |
| `workspace/qdrant/` | Embedded vector index | Optional | Yes, but you must re-ingest (costs money) |
| `workspace/cache/` | Cached generations and judge verdicts | Recommended | Yes, but re-runs then re-bill |
| `workspace/data/` | Uploaded corpora and evalsets | **Yes** | No, unless you have the originals |
| `.env` | Keys | **No**, never |, |

Backing up is a file copy; there is no database:

```bash
tar czf eval-backup-$(date +%F).tar.gz workspace/traces workspace/data
```

The trace store is append-only, so a backup taken mid-run is consistent up to
the last checkpoint. To fold many parts into one portable file:

```python
from harness.store.store import TraceStore
TraceStore("workspace/traces").compact()
```

---

## Upgrading an existing install

```bash
git pull
pip install -r requirements.txt
pytest                                   # 964 tests, offline, free
python main.py validate --profile <your_profile>
```

Two migration notes:

- **Old results are picked up automatically.** Earlier versions wrote a single
  `traces.parquet`; the store is now an append-only directory. A legacy file
  sitting beside the directory is adopted and read transparently, so your
  history stays visible. Nothing to do.
- **Profiles are validated on load** and now reject mistakes that used to pass
  silently, a positive weight on a lower-is-better metric, weights on metrics
  the profile never collects, typos in `active_metrics`. `validate` reports every
  problem at once.

---

## Operational limits

Know these before you scale up:

- **Embedded Qdrant is single-process.** The folder is locked by whoever opens
  it. Do not run the app and a CLI ingest against the same `workspace/qdrant`
  at the same time, use a Qdrant server if you need concurrent access.
- **The app runs one heavy job at a time**, enforced. It is a single-user tool;
  two people clicking Run will queue behind each other at best.
- **`max_workers` is throughput-lane concurrency.** Raising it past your
  provider's rate limit makes runs slower, not faster, because everything ends
  up in backoff. Prefer a per-provider `rate_limit` in `configs/models.yaml`.
- **Latency numbers require the latency lane.** It runs serially and uncached by
  design; timings taken from the concurrent lane measure your own queue.
- **Memory** is roughly 2–3 GB during a run. Ingestion is the spiky part and is
  batched to keep it flat.

---

## Health checks

```bash
# Config, dataset and provider routing, before spending anything
python main.py validate --profile regulated_qa

# What a run would cost, judge calls included
python main.py estimate --profile regulated_qa

# What has been recorded so far
python main.py runs

# The offline suite: no key, no network, no cost
pytest

# The web UI: the landing answers 200, and the session probe answers
# {"user": null} without a cookie; both are free and need no sign-in
curl -fsS -o /dev/null http://127.0.0.1:8010/
curl -fsS http://127.0.0.1:8010/api/auth/me
```

For a monitored deployment, `validate` is the right liveness probe: it exits
non-zero when a provider is unreachable, a key is missing, or a profile is
invalid, without making a billable call.

---

## Production checklist

Before letting anyone else touch it:

- [ ] `pytest` passes on the target machine
- [ ] `python main.py validate --profile <p>` is clean
- [ ] Keys are in the environment or `.env`, **not** in the image or a commit
- [ ] `budget_usd` is set in `configs/run.yaml`
- [ ] `configs/pricing.yaml` refreshed: `validate` warns when it is stale, and
      stale prices mis-rank models rather than just mis-stating dollars
- [ ] `HARNESS_PILOT_PASSWORD` is set; the console no longer says the default is in use
- [ ] `python main.py serve` runs with `--budget`, bound to loopback, behind TLS
- [ ] Only the people who own the spend know to pick the Assurance Lead role
- [ ] The image, if any, was built from `.dockerignore`'s view of the tree: no `.env`, no `workspace/`
- [ ] `workspace/traces/` and `workspace/data/` are backed up
- [ ] A known-good `run_id` is recorded somewhere as the gate's baseline
- [ ] The CI gate has `--budget` and `--max-cost` set
