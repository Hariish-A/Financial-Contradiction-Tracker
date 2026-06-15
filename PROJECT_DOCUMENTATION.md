# Project Documentation: Financial Guidance Contradiction Tracker (ContraGuard)

This document provides a comprehensive, milestone-by-milestone technical deep dive into the **Financial Guidance Contradiction Tracker (ContraGuard)**. It covers the core architectures, mathematical scoring formulas, database schemas, natural language processing pipelines, and machine learning models employed in this system.

---

## 📖 Executive Summary & Core Value Proposition

In the financial markets, tracking corporate earnings call statements is traditionally done using keyword search (e.g., Bloomberg terminals looking for changes in specific words). However, keyword tracking fails to detect **reasoning-level alterations, structural goalpost-shifting, and silent omissions of commitments** made by corporate executives (CEOs, CFOs, and Managing Directors) across quarters.

**ContraGuard** solves this by:
1. **Ingesting raw data** from Indian corporate filings (BSE India) and structured financials (Screener.in).
2. **Diarizing speaker transcripts** to isolate CEO, CFO, and MD commentary.
3. **Extracting and classifying guidance statements** into quantitative, qualitative, hedged, and deflection claims using sentiment models and heuristics.
4. **Detecting three layers of contradictions** across consecutive quarters:
   - **HARD Contradictions**: Explicit logical reversals detected via Natural Language Inference (NLI).
   - **SOFT Contradictions**: Tactical shifts detected via a hybrid metric composed of topic similarity, sentiment flips, and hedge tone escalation.
   - **OMISSION Contradictions**: Silent dropout of key topics previously highlighted for multiple quarters.
5. **Calculating an Executive Credibility Score** (0–100) by cross-referencing forward-looking numeric promises against actual financial outcomes scraped from Screener.in.
6. **Surfacing findings** in a premium Streamlit dashboard with timelines, risk tiers, and real-time semantic search.

---

## 📐 System Architecture

The project is structured as a 5-layer pipeline:

```mermaid
graph TD
    A[BSE PDF Announcements & Screener.in] -->|Milestone 1: Ingestion| B[(SQLite Database: tracker.db)]
    B -->|Milestone 2: Diarizer & spaCy Sentencizer| C[Executive Statements]
    C -->|Milestone 2: FinBERT & Regex Classifier| D[Classified Statements & Sentiment]
    D -->|Milestone 3: ST all-mpnet-base-v2 & FAISS| E[Semantic Search & Retrieval]
    E -->|Milestone 3 & 4: NLI DeBERTa & Hybrid Scorer| F[HARD, SOFT, & OMISSION Contradictions]
    D -->|Milestone 5: Regex Parser| G[Extracted Numeric Predictions]
    G -->|Milestone 5: Actuals Verification| H[Direction Accuracy / Magnitude Error]
    F & H -->|Milestone 5: Scorer Engine| I[Executive Credibility Score [0-100]]
    I -->|Milestone 6: Streamlit| J[ContraGuard Interactive Dashboard]
```

---

## 🛠️ Technology Stack Summary

| Component | Tool / Model | Specific Role |
| :--- | :--- | :--- |
| **PDF Parsing** | PyMuPDF (`fitz`) | High-speed text extraction from transcript PDFs with layout sorting. |
| **Web Scraping** | `requests` + `BeautifulSoup` (`lxml`) | Fetches announcements from BSE India and quarterly results tables from Screener.in. |
| **NLP Pipeline** | `spaCy` (`en_core_web_sm`) | Sentence boundary detection (Sentencizer) and noun phrase extraction for topic mapping. |
| **Sentence Embedding** | `sentence-transformers/all-mpnet-base-v2` | Computes 768-dimensional L2-normalized vectors representing statement semantics. |
| **Vector Database** | `FAISS` (`IndexFlatIP`) | Conducts sub-millisecond cosine similarity searches over an executive's historical statements. |
| **NLI Model** | `cross-encoder/nli-deberta-v3-base` | Computes contradiction, neutral, and entailment probabilities between statement pairs. |
| **Sentiment Analysis** | `ProsusAI/finbert` | Classifies financial statement sentiment (positive, negative, neutral) and returns confidence scores. |
| **Storage Engine** | SQLite + DuckDB | SQLite handles transactional reads/writes; DuckDB executes fast OLAP cross-quarter analytical queries. |
| **User Interface** | Streamlit + Plotly | Surfaced as a responsive, dark-themed dashboard. |

---

## 📅 Milestone-by-Milestone Technical Breakdown

### 📂 Milestone 1: Ingestion & Parser Layer
Located in: [ingestion/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/ingestion/)

This layer is responsible for gathering raw data, cleaning it, extracting text, attributing dialogues, and populating the baseline SQLite database.

1. **BSE Scraper ([bse_scraper.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/ingestion/bse_scraper.py))**:
   - Queries BSE India's public JSON API (`AnnSubCategoryGetData`) using company BSE codes (e.g., `500209` for Infosys) and category code `57` (Investor Presentation/Earnings Call transcript).
   - Dynamically checks attachments at `AttachLive`. If a document returns a `404`, it fails over to the historical attachment directory `AttachHis`.
   - Normalizes calendar dates to Indian Fiscal quarters using a calendar map. (Note: Indian Fiscal Year runs April to March; e.g., June 2023 is Q1FY24, and March 2024 is Q4FY24).
2. **Screener Scraper ([screener_scraper.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/ingestion/screener_scraper.py))**:
   - Scrapes `https://www.screener.in/company/{ticker}/consolidated/`. If the consolidated page fails, it falls back to the standalone page.
   - Extracts the "Quarterly Results" HTML table, converting figures (Sales, Margin %, Net Profit, EPS) into a clean Pandas DataFrame.
   - Normalizes Screener's columns (e.g. `Mar 2024` $\rightarrow$ `Q4FY24`) using calendar-year-end rules.
3. **PDF Extractor ([pdf_extractor.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/ingestion/pdf_extractor.py))**:
   - Opens documents using PyMuPDF (`fitz`).
   - Uses `page.get_text("text", sort=True)` to ensure multi-column text blocks are read in correct reading order (left-to-right, top-to-bottom) rather than raw PDF string coordinates.
   - Applies precompiled regular expressions to strip boilerplate headers/footers, page numbers (`Page X of Y`), and analyst house watermarks (e.g., "Motilal Oswal", "Strictly Confidential").
4. **Speaker Diarizer ([diarizer.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/extraction/diarizer.py))**:
   - Segments raw text into speaker turns. It uses regular expressions targeting uppercase name structures followed by roles and colons (e.g., `Rajesh Kumar - CFO:` or `Srinivasan Vaidyanathan (CFO):`).
   - Resolves speaker roles using two fallbacks:
     - **Direct regex capture** from dialogue tags.
     - **Context window search**: If the role is missing, it searches a 100-character window around the first mention of the speaker's name in the transcript for executive keywords like "CEO", "CFO", "MD", "Managing Director", "Chief Financial", or "Chief Executive".
   - Upserts matching executive records to the `executives` database table and stores text blocks under target roles (`CEO`, `CFO`, `MD`).

---

### 🗣️ Milestone 2: Statement & Executive Extraction Layer
Located in: [extraction/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/extraction/)

This layer takes the raw text blocks attributed to targeted executives and structures them into individual, classified statements.

1. **Statement Extractor ([statement_extractor.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/extraction/statement_extractor.py))**:
   - Leverages `spaCy`'s pipeline. To maximize performance, it disables the Named Entity Recognition (`ner`) and Text Classifier (`textcat`) models, enabling only the `sentencizer` pipe.
   - Normalizes text spacing and filters out dialogues under 15 characters (e.g., "Yes.", "Thank you.", "Alright.") to eliminate conversation filler.
2. **Statement Classifier ([classifier.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/extraction/classifier.py))**:
   - Classifies each sentence into one of five categories using rule-based heuristics:
     - `DEFLECTION`: Sentences containing deflection terms (e.g., "too early", "cannot comment", "wait and watch", "not in a position").
     - `QUANTITATIVE_GUIDANCE`: Sentences containing guidance keywords (e.g., "expect", "target", "guidance") and numbers/percentages (e.g., `18%`, `basis points`, `bps`, `crore`).
     - `QUALITATIVE_GUIDANCE`: Sentences containing guidance keywords but lacking numeric metrics.
     - `HEDGED`: Sentences containing tone-dampening hedge words (e.g., "cautious", "optimistic", "headwind", "difficult", "stable").
     - `FACTUAL_CLAIM`: Historical claims or assertions lacking forward guidance signals.
   - Analyzes sentiment using the `ProsusAI/finbert` pipeline (configured for GPU if available, else defaulting to CPU), returning a label (`positive`, `negative`, `neutral`) and its confidence score.

---

### 🔍 Milestone 3: Semantic Retrieval & Hard Contradictions
Located in: [contradiction/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/)

This layer implements semantic matching and evaluates logical contradictions.

1. **Sentence Embeddings & FAISS Index ([embeddings.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/embeddings.py))**:
   - Computes 768-dimensional sentence representations using the HuggingFace model `sentence-transformers/all-mpnet-base-v2`.
   - L2-normalizes all vectors:
     $$\hat{v} = \frac{v}{\|v\|_2}$$
   - Initializes a FAISS index (`faiss.IndexFlatIP`) per executive. Because vectors are L2-normalized, the inner product (IP) computed by FAISS is equivalent to cosine similarity:
     $$\text{Cosine Similarity} = \langle \hat{v}_a, \hat{v}_b \rangle$$
2. **NLI Scorer ([nli_scorer.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/nli_scorer.py))**:
   - Uses the DeBERTa Cross-Encoder model `cross-encoder/nli-deberta-v3-base`.
   - Feeds the model statement pairs $(A, B)$ to compute raw logits. These are converted to probabilities using softmax:
     $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$
   - Maps the output layer indices to `contradiction`, `neutral`, and `entailment`.
   - **Hard Contradiction Rule**: If the pair's contradiction probability exceeds `HARD_CONTRADICTION_THRESHOLD` (0.5), it is flagged as a HARD contradiction.
     - *Example:*
       - **Statement A (Q2):** *"We expect 18% revenue growth in the next quarter."*
       - **Statement B (Q3):** *"We are revising our guidance to 8% for the quarter."*
       - **DeBERTa Verdict:** `CONTRADICTION` (Probability: 0.98).

---

### ⚖️ Milestone 4: Soft Contradictions & Omission Detection
Located in: [contradiction/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/)

This layer detects non-obvious, strategic contradictions where executives shift their tone, sentiment, or topic coverage without direct logical conflict.

1. **Soft Contradiction Detector ([soft_detector.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/soft_detector.py))**:
   - Measures shifts in strategy by calculating a weighted score across three signals:
     - **Topic Similarity ($S_{\text{topic}}$)**: Cosine similarity of the two statement embeddings. Requires a similarity $\ge 0.6$ to ensure statements cover the same topic.
     - **Sentiment Flip ($S_{\text{sent}}$)**: Calculates the polarization shift:
       - Flipped from positive $\leftrightarrow$ negative: $1.0$
       - Flipped from positive/negative $\leftrightarrow$ neutral: $0.5$
       - Unchanged sentiment: $0.0$
     - **Hedge Escalation ($S_{\text{hedge}}$)**: Measures increased caution by scanning text for keywords in the `HEDGE_SCALE` map. Matches are evaluated on a confidence scale (e.g., "confident" = 1.0, "cautious" = 0.4, "headwind" = 0.2, "difficult" = 0.1).
       $$S_{\text{hedge}} = \max(0.0, H_{\text{score\_A}} - H_{\text{score\_B}})$$
     - **Composite Formula**:
       $$\text{Soft Score} = 0.4 \times S_{\text{topic}} + 0.4 \times S_{\text{sent}} + 0.2 \times S_{\text{hedge}}$$
     - **Threshold Rule**: Pairs with a composite score exceeding `SOFT_CONTRADICTION_THRESHOLD` (0.6) are flagged as a `SOFT` contradiction.
2. **Omission Contradiction Detector ([omission_detector.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/contradiction/omission_detector.py))**:
   - Extracts noun phrases using `spaCy` noun chunks, filtering out determiners, pronouns, and general stop words.
   - Traces topic frequencies per executive chronologically.
   - **Omission Rule**: If a topic is mentioned in $N$ consecutive quarters (default $N = 3$) but is dropped in the subsequent quarter, it is flagged as an `OMISSION` contradiction.
     - *Example:* If an executive guides on "rural segment" in Q1, Q2, and Q3, but fails to mention "rural" in any Q4 statement, the topic is marked as omitted.

---

### 📈 Milestone 5: Credibility Scorer & Verification Pipeline
Located in: [credibility/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/credibility/)

This layer extracts quantitative claims, tracks their accuracy against scraped outcomes, and updates the executive's credit rating.

1. **Prediction Extractor ([scorer.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/credibility/scorer.py))**:
   - Uses precompiled regular expressions to extract numeric values (percentages, currency, basis points) from guidance statements.
   - Maps surrounding tokens to canonical metrics (e.g. "sales" $\rightarrow$ `revenue_growth`, "ebitda" $\rightarrow$ `ebitda_margin`, "hiring" $\rightarrow$ `headcount`).
   - Evaluates surrounding verbs to classify the predicted direction (`up`, `down`, or `stable`).
2. **Credibility Scorer ([scorer.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/credibility/scorer.py))**:
   - Gathers all contradiction counts and verified predictions for a given executive.
   - **Verification Math**:
     - Upward calls are marked correct if Actual > Predicted.
     - Downward calls are marked correct if Actual < Predicted.
     - Stable calls are marked correct if the outcome falls within a 5% margin of the prediction:
       $$\left|\text{Actual} - \text{Predicted}\right| \le 0.05 \times \left|\text{Predicted}\right|$$
   - **Composite Score Formula**:
     - Starts with a base score of 100 and applies weights for contradictions and predictions:
       $$\text{Score} = 100 - (N_{\text{hard}} \times 20) - (N_{\text{soft}} \times 10) - (N_{\text{omit}} \times 5) + (N_{\text{correct}} \times 10) - (N_{\text{wrong}} \times 10)$$
     - Clamps final scores to the range $[0, 100]$.
   - **Accuracy score (from predictions)**:
     - Measures prediction error:
       $$\text{Pct Error} = \frac{|\text{Actual} - \text{Predicted}|}{|\text{Predicted}|} \times 100$$
       $$\text{Accuracy Score} = (\text{Directional Accuracy \%}) \times 0.70 + (\max(0, 100 - \text{Avg Pct Error})) \times 0.30$$
   - **Risk Tiers**:
     - $\ge 70$: **LOW RISK** (Green)
     - $50 - 69$: **MEDIUM RISK** (Orange)
     - $< 50$: **HIGH RISK** (Red)

---

### 🖥️ Milestone 6: Interactive Dashboard
Located in: [dashboard/](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/dashboard/)

Surfaces pipeline data through a responsive Streamlit UI.

1. **Dark Mode UI ([app.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/dashboard/app.py))**:
   - Custom CSS styling creates a dark-themed corporate design. It imports the *Inter* and *JetBrains Mono* fonts, configures card borders and gradients, and formats buttons and metric boxes.
2. **Data Caching ([data_fetcher.py](file:///d:/Official/Projects/Financial%20Contradiction%20Tracker/dashboard/data_fetcher.py))**:
   - Wraps database calls in `@st.cache_data(ttl=60)` to prevent dashboard lag from recurrent SQL queries.
   - Exposes clear mutation endpoints (e.g. `verify_prediction()`) that call `st.cache_data.clear()` to force-refresh data on update.
3. **Timeline & Semantic Search**:
   - Embeds Plotly charts showing executive risk ratings.
   - Provides side-by-side timeline panels for comparing statement pairs.
   - Connects to the FAISS index to run semantic search queries.

---

## 🗄️ Database Architecture & Schema

### SQLite Schema DDL
The relational SQLite database contains five tables:

```sql
-- Track target organizations
CREATE TABLE companies (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT    NOT NULL,
    bse_code TEXT    NOT NULL UNIQUE,
    sector   TEXT
);

-- Track corporate executives
CREATE TABLE executives (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    role       TEXT    NOT NULL,          -- CEO, CFO, MD
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    UNIQUE(name, company_id)
);

-- Track PDF source filings and raw text
CREATE TABLE transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    quarter     TEXT    NOT NULL,         -- e.g., Q1FY24
    year        INTEGER NOT NULL,
    source_url  TEXT,
    pdf_path    TEXT,
    raw_text    TEXT,
    processed   INTEGER DEFAULT 0,        -- 0=raw, 1=extracted
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- Store diarized, classified sentences with embeddings
CREATE TABLE statements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    executive_id    INTEGER NOT NULL REFERENCES executives(id),
    company_id      INTEGER NOT NULL REFERENCES companies(id),
    transcript_id   INTEGER REFERENCES transcripts(id),
    quarter         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    statement_type  TEXT,                -- QUANTITATIVE_GUIDANCE, QUALITATIVE_GUIDANCE, etc.
    sentiment       TEXT,                -- positive, negative, neutral
    sentiment_score REAL,
    embedding       BLOB,                -- Serialized numpy float32 array
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Store contradiction pairs
CREATE TABLE contradictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_a_id     INTEGER NOT NULL REFERENCES statements(id),
    statement_b_id     INTEGER NOT NULL REFERENCES statements(id),
    contradiction_type TEXT    NOT NULL,  -- HARD, SOFT, OMISSION
    score              REAL    NOT NULL,
    details            TEXT,              -- JSON metadata
    reviewed           INTEGER DEFAULT 0,
    created_at         TEXT DEFAULT (datetime('now'))
);

-- Track numeric predictions and verified actual outcomes
CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    executive_id    INTEGER NOT NULL REFERENCES executives(id),
    statement_id    INTEGER REFERENCES statements(id),
    quarter         TEXT    NOT NULL,
    metric          TEXT    NOT NULL,    -- revenue_growth, margin, etc.
    predicted_value REAL,
    direction       TEXT,                -- up, down, stable
    actual_value    REAL,
    outcome_quarter TEXT,
    verified        INTEGER DEFAULT 0
);

-- Database indexes for optimized lookup
CREATE INDEX idx_statements_exec   ON statements(executive_id);
CREATE INDEX idx_statements_quarter ON statements(quarter, year);
CREATE INDEX idx_contradictions_type ON contradictions(contradiction_type);
```

### DuckDB Integration
DuckDB is integrated as an OLAP bridge for complex cross-quarter analytics. It connects directly to the active SQLite file:

```python
import duckdb

def run_duckdb_query(sql_query: str):
    con = duckdb.connect()
    # Attach the SQLite file in read-only mode
    con.execute("ATTACH 'data/tracker.db' AS tracker (TYPE sqlite, READ_ONLY)")
    return con.execute(sql_query).fetchdf()
```

This configuration enables sub-millisecond aggregation queries across millions of rows without impacting transactional performance.

---

## 🔬 Machine Learning Model Details

This system combines several NLP models, each selected for its target performance within the pipeline:

### 1. `sentence-transformers/all-mpnet-base-v2`
- **Role**: Computes sentence embeddings for semantic retrieval.
- **Why Chosen**: Maps sentences to a dense vector space where distance represents semantic similarity. It performs consistently on the Sentence-BERT semantic similarity benchmark.
- **Configuration**: Produces 768-dimensional float32 arrays. L2-normalized upon extraction, permitting cosine similarity calculations via standard dot products.

### 2. `cross-encoder/nli-deberta-v3-base`
- **Role**: Natural Language Inference (NLI) classification.
- **Why Chosen**: Standard dual-encoder models evaluate statements separately, which can miss complex logical contradictions. Cross-encoders feed the sentence pair into the transformer layer simultaneously, allowing full attention over all token relationships. This configuration produces more accurate logical classifications.
- **Configuration**: Evaluates $(A, B)$ pairs to classify relation probabilities (contradiction, entailment, neutral) over a sequence length of 512 tokens.

### 3. `ProsusAI/finbert`
- **Role**: Sentiment analysis.
- **Why Chosen**: General-purpose sentiment classifiers often misinterpret corporate terminology (e.g., flagging "headwinds" or "reduced margins" as neutral or positive). FinBERT is a BERT model pre-trained on the Financial PhraseBank corpus, allowing it to correctly classify financial expressions.
- **Configuration**: Outputs probability scores for positive, negative, and neutral sentiment.

---

## 🏃 Execution & Deployment Playbook

Ensure your virtual environment is activated before running pipeline scripts:
```powershell
.\venv\Scripts\activate
```

### Step 1: Initialise & Run Ingestion
Downloads PDFs from BSE India, scrapes financial figures from Screener.in, and initializes the database:
```powershell
# Ingest all 5 companies and 8 quarters
python run_ingestion.py

# Ingest only a single company (e.g., Infosys) and skip Screener.in to save time
python run_ingestion.py --company 500209 --skip-screener
```

### Step 2: Run Extraction Pipeline
Diarizes raw transcripts, extracts sentences, and classifies their statements and sentiment:
```powershell
# Process all unprocessed transcripts
python run_extraction.py

# Test on a limited subset of transcripts
python run_extraction.py --limit 2
```

### Step 3: Run Contradiction Engine
Computes statement embeddings and scans historical records for contradictions:
```powershell
# 1. Compute and backfill embeddings for all statements in the database
python run_contradiction.py --backfill

# 2. Run the NLI model on verification test pairs
python run_contradiction.py --test-cases

# 3. Run the full contradiction engine (HARD, SOFT, OMISSION)
python run_contradiction.py --run-pipeline

# 4. Filter pipeline scans to a single executive
python run_contradiction.py --run-pipeline --filter-exec 1
```

### Step 4: Run Credibility Engine
Extracts predictions and calculates executive credibility:
```powershell
# 1. Extract numeric predictions from guidance statements
python run_credibility.py --extract-predictions

# 2. Review all extracted predictions
python run_credibility.py --list-predictions

# 3. Verify a prediction by recording an actual outcome (e.g., 14.5% margin)
python run_credibility.py --verify --pred-id 3 --actual 14.5

# 4. Calculate and output credibility scores for all executives
python run_credibility.py --score
```

### Step 5: Start Streamlit Dashboard
Runs the web dashboard to visualize scores, timelines, and run semantic queries:
```powershell
streamlit run dashboard/app.py
```
The interface will open at `http://localhost:8501`.
