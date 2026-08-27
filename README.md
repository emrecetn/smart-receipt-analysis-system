# Smart Receipt Analysis System

[![Tests](https://github.com/emrecetn/smart-receipt-analysis-system/actions/workflows/tests.yml/badge.svg)](https://github.com/emrecetn/smart-receipt-analysis-system/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered receipt/invoice data extraction API, built as a university capstone project. It combines a custom-trained **YOLOv8** model with **GPT-4o Vision** behind a **FastAPI** backend to turn a photo of a receipt into structured, validated JSON.

> Built as an academic capstone project — a portfolio piece, not a production/commercial service.

## How it works

```
Photo upload
   │
   ▼
YOLOv8  ──────────────►  detects and crops the receipt out of the background
   │
   ▼
GPT-4o Vision  ────────►  reads the cropped image and extracts structured fields
   │
   ▼
Pydantic validation  ──►  guarantees the API always returns a well-typed JSON shape
```

Cropping with YOLOv8 before calling the vision model improves extraction accuracy (less background noise) and reduces image size/token cost.

## Features

- **Structured extraction** from a receipt/invoice photo: merchant name, tax ID (VKN/TCKN), date, receipt number, total amount, and a full VAT breakdown by rate.
- **Broad format support**: JPEG, PNG, WEBP, and HEIC (iPhone's default photo format).
- **Per-customer API keys**: generated client-side with cryptographically secure randomness, sent to the server, and stored only as a SHA-256 hash — the raw key is never persisted anywhere and is shown to the user exactly once.
- **Supabase-backed auth**: user accounts and API key management via Supabase (PostgreSQL + Auth).
- **Typed, validated API contract**: every endpoint response is a Pydantic model, so malformed AI output is caught and turned into a clean `502`, never a crash.
- **Automated tests**: 28 pytest tests covering authentication, file validation, and the full YOLO → GPT-4o pipeline (mocked, no external calls in CI).
- **CI**: GitHub Actions runs the full test suite on every push and pull request.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI, Pydantic v2, Uvicorn |
| Computer vision | Ultralytics YOLOv8 (custom-trained receipt detector), OpenCV, Pillow + pillow-heif |
| AI extraction | OpenAI GPT-4o Vision |
| Database / Auth | Supabase (PostgreSQL + Auth) |
| Frontend | HTML, vanilla JavaScript, Tailwind CSS |
| Testing | pytest, pytest-mock |
| CI | GitHub Actions |

## Getting started

### Prerequisites

- Python 3.12+
- A [Supabase](https://supabase.com) project (project URL + service role key)
- An [OpenAI](https://platform.openai.com) API key with GPT-4o access

### Installation

```bash
git clone https://github.com/emrecetn/smart-receipt-analysis-system.git
cd smart-receipt-analysis-system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your own credentials:

```bash
cp .env.example .env
```

Run the server:

```bash
py -m uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser. Interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests mock both YOLO and OpenAI, so they run without any external API calls or a real Supabase connection required for most cases.

## API overview

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | — | Service health check (model loaded, DB connected) |
| `/api/v1/extract-receipt` | POST | `X-API-Key` header | Upload a receipt image, get back structured JSON |

A demo key (`sk_test_demo123456789`) is available out of the box for quick testing without registering an account.

## Project structure

```
.
├── main.py                  # FastAPI application (API, auth, YOLO + GPT-4o pipeline)
├── index.html                # Frontend (landing page + developer dashboard)
├── tests/                    # pytest suite
├── ml/                        # Model training assets
│   ├── train.py               # YOLOv8 training script
│   ├── dataset/                # data.yaml / classes.txt (images/labels are gitignored)
│   └── runs/detect/.../weights/best.pt   # trained model weights
├── legacy/                    # earlier OCR/LLM experiments, kept for reference
└── .github/workflows/tests.yml   # CI pipeline
```

## Security notes

API keys are generated in the browser using `crypto.getRandomValues` (not `Math.random`), then hashed with SHA-256 before being sent to the server — the database only ever stores the hash, never the raw key. Generating a new key immediately invalidates the previous one.

## License

MIT — see [LICENSE](LICENSE).
