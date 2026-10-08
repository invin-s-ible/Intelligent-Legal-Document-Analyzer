# Intelligent Legal Document Analyzer

An AI/NLP web app that reads a legal document and extracts parties, dates, amounts, key clauses, obligations and risky wording, then writes a short summary.

## Tech stack (as per project brief)
| Brief item | Where it is used |
|---|---|
| Python | whole backend |
| Pandas | `data/legal_clauses.csv` dataset loading, notebook tables |
| NLTK | sentence splitting (`analyzer/pipeline.py`) |
| spaCy | named entities (organisations, places, money) |
| scikit-learn | TF-IDF + Logistic Regression clause classifier, TF-IDF extractive summary |
| Hugging Face Transformers | optional abstractive summary (`USE_TRANSFORMERS=1`) |
| Flask | web server and JSON API (`app.py`) |
| Jupyter / Colab | `notebooks/legal_analyzer_demo.ipynb` |
| Git / GitHub | see below |

## Project layout
```
app.py                   Flask server (/, /api/analyze, /api/extract)
analyzer/pipeline.py     NLP pipeline (entities, clauses, obligations, risks, summary)
analyzer/readers.py      .txt / .pdf / .docx text extraction
data/legal_clauses.csv   labelled sample clause dataset (13 clause types)
data/sample_contract.txt sample agreement for the demo
train_classifier.py      train + evaluate + save the clause classifier
templates/index.html     3D landing page and results UI
notebooks/               Colab/Jupyter demo
tests/                   pytest test
```

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python train_classifier.py         # optional: saves models/clause_clf.joblib
python app.py                      # open http://127.0.0.1:5000
```
Optional Hugging Face summary: `pip install transformers torch`, then set `USE_TRANSFORMERS=1` before `python app.py` (Windows PowerShell: `$env:USE_TRANSFORMERS="1"`). The first run downloads the model.

Run the test: `python -m pytest -q tests`

## Demo video checklist
1. Open the site and show the 3D hero.
2. Upload `data/sample_contract.txt` (or a PDF/DOCX), or click "Load sample".
3. Show the pipeline steps, risk score and summary.
4. Open each tab: Key clauses, Entities, Obligations, Attention areas.
5. Show the highlighted document and click "Copy report".

## Git / GitHub
```bash
git init
git add .
git commit -m "Intelligent Legal Document Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/legal-doc-analyzer.git
git push -u origin main
```

## Limitations
- The clause dataset is small (52 sentences), so cross-validated accuracy is about 46%. Add more labelled clauses to `data/legal_clauses.csv` and rerun `train_classifier.py` to improve it.
- Entity and risk rules are tuned for English commercial contracts.
- This is a study project, not legal advice.
