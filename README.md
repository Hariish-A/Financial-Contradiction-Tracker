# Financial Guidance Contradiction Tracker

> **"Bloomberg tracks keyword changes. This system detects reasoning-level contradictions and scores whether management can be trusted."**

A system that ingests Indian company earnings-call transcripts, extracts forward-guidance statements by CFOs/CEOs, detects hard and soft contradictions across quarters, and scores executive credibility by comparing predictions to actual outcomes — all surfaced in a Streamlit dashboard.

---

## System Architecture

```
DATA INGESTION → EXTRACTION → CONTRADICTION ENGINE → CREDIBILITY SCORER → DASHBOARD
```

| Layer | Module | Status |
|---|---|---|
| 1 — Data Ingestion | `ingestion/` | ✅ Week 1 |
| 2 — Extraction | `extraction/` | ✅ Week 2 |
| 3 — Contradiction Engine | `contradiction/` | 🚧 Week 3–4 (Milestone 3 ✅) |
| 4 — Credibility Scorer | `credibility/` | 🔜 Week 5 |
| 5 — Dashboard | `dashboard/` | 🔜 Week 6 |

---

## Project Structure

```
financial-contradiction-tracker/
├── ingestion/
│   ├── bse_scraper.py          # Scrapes BSE for transcript PDF links + downloads
│   ├── pdf_extractor.py        # PyMuPDF text extraction + noise cleaning
│   ├── screener_scraper.py     # Pulls structured quarterly financials from Screener.in
│   └── orchestrator.py         # Wires all 3 scrapers for 5 companies × 8 quarters
├── extraction/
│   ├── diarizer.py             # (Week 2) Speaker attribution
│   ├── statement_extractor.py  # (Week 2) Sentence-level claim extraction
│   └── classifier.py           # (Week 2) FinBERT guidance type classifier
├── contradiction/
│   ├── embeddings.py           # (Week 3) FAISS index + FinancialBERT
│   ├── nli_scorer.py           # (Week 3) DeBERTa hard contradiction
│   ├── soft_detector.py        # (Week 4) Topic + sentiment + hedge scoring
│   └── omission_detector.py    # (Week 4) Topic dropout across quarters
├── credibility/
│   └── scorer.py               # (Week 5) Prediction vs actual tracker
├── storage/
│   └── database.py             # SQLite schema + CRUD + DuckDB analytics bridge
├── dashboard/
│   └── app.py                  # (Week 6) Streamlit dashboard
├── data/
│   ├── transcripts/            # Raw PDFs per company per quarter (git-ignored)
│   └── tracker.db              # SQLite database (git-ignored)
├── logs/                       # Runtime logs (git-ignored)
├── config.py                   # All constants, company list, thresholds
├── run_ingestion.py            # CLI entry point for Week 1
├── run_extraction.py           # CLI entry point for Week 2
├── run_contradiction.py        # CLI entry point for Week 3 (Milestone 3)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Initialise the database
```bash
python run_ingestion.py --init-only
```

### 3. Run full ingestion (5 companies, 8 quarters)
```bash
python run_ingestion.py
```

### 4. Single company (faster for testing)
```bash
# Infosys only, skip Screener.in
python run_ingestion.py --company 500209 --skip-screener
```

### 5. Force re-download
```bash
python run_ingestion.py --overwrite
```

---

## Target Companies (Week 1)

| Company | BSE Code | NSE Ticker | Sector |
|---|---|---|---|
| Reliance Industries | 500325 | RELIANCE | Conglomerate |
| Infosys | 500209 | INFY | IT Services |
| HDFC Bank | 500180 | HDFCBANK | Banking |
| TCS | 532540 | TCS | IT Services |
| Wipro | 507685 | WIPRO | IT Services |

Target quarters: **Q1FY23 → Q4FY24** (8 quarters per company)

---

## Tech Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Web scraping | `requests` + `BeautifulSoup` |
| Speaker diarization | Regex + spaCy NER |
| Sentence embedding | FinancialBERT (FAISS) |
| NLI model | DeBERTa-v3 cross-encoder |
| Sentiment | `ProsusAI/finbert` |
| Storage | SQLite + DuckDB |
| Dashboard | Streamlit + Plotly |

---

## Build Milestones

| Week | Deliverable | Status |
|---|---|---|
| 1 | Scraper working, 5 companies, 8 quarters of transcripts | ✅ |
| 2 | Speaker diarization + statement extractor + classifier | ✅ |
| 3 | FAISS index + NLI contradiction scorer | ✅ |
| 4 | Soft contradiction detector + hedge escalation | 🔜 |
| 5 | Credibility scorer tracking 3 executives across 2 years | 🔜 |
| 6 | Streamlit dashboard: timeline + scorecard + search + PDF export | 🔜 |

---

## Database Schema

Five tables: `companies → executives → statements → contradictions → predictions`

SQLite for reads/writes · DuckDB for cross-quarter analytics queries.

---

## Recruiter Pitch

*"I built a system that tracks forward guidance contradictions in earnings calls — detecting both hard logical contradictions and soft sentiment reversals using NLI + hedge escalation scoring, then scoring executive credibility by comparing what was promised vs what was delivered across 8 quarters."*

---

## Running Instructions

Ensure your virtual environment is activated before running the scripts:
```powershell
.\venv\Scripts\activate
```

### Week 1: Data Ingestion
This step downloads all earnings call PDFs from BSE India and financial metrics from Screener.in, then saves the raw text into the SQLite database.
```powershell
python run_ingestion.py
```
*(Tip: Use `python run_ingestion.py --help` for options like running a single company)*

### Week 2: Extraction Layer
This step processes the raw transcript text. It diarizes the text by speaker, extracts sentences, and uses FinBERT and regex rules to classify guidance types and sentiment.
```powershell
python run_extraction.py
```
*(Tip: Use `python run_extraction.py --limit 2` to test the pipeline on a subset of transcripts)*

### Week 3: Contradiction Engine Foundation
This step computes vector embeddings for all statements in the database and runs semantic similarity searches or NLI verification test cases.

#### 1. Backfill Statement Embeddings
Compute and store 768-dimensional embeddings for all extracted statements in the SQLite database:
```powershell
python run_contradiction.py --backfill
```

#### 2. Run Verification Test Cases
Verify the NLI DeBERTa scorer on standard logical contradiction, sentiment flip, hedge escalation, and entailment pairs:
```powershell
python run_contradiction.py --test-cases
```

#### 3. Semantic Search Query
Search an executive's statements semantically using their dynamically constructed FAISS index:
```powershell
python run_contradiction.py --exec-id <executive_id> --query "<query_text>"
```
*(Example: `python run_contradiction.py --exec-id 1 --query "growth"`)*
