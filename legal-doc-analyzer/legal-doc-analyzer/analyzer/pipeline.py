"""Intelligent Legal Document Analyzer - NLP pipeline.

Stages: clean -> sentence split (NLTK) -> entities (spaCy + regex)
-> clause classification (scikit-learn TF-IDF + Logistic Regression, trained on a Pandas dataset)
-> obligations -> risk rules -> summary (TF-IDF extractive, optional Hugging Face abstractive).
"""
import os
import re
import joblib
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE, "data", "legal_clauses.csv")
MODEL_PATH = os.path.join(BASE, "models", "clause_clf.joblib")

M = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
DATE = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + M + r",?\s+\d{4}\b|\b" + M +
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)
MONEY = re.compile(r"(?:₹|\$|€|£|Rs\.?|INR|USD)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:lakh|crore|million|billion))?|\b\d+(?:\.\d+)?\s?%", re.I)
TERM = re.compile(r"\b\d+(?:-|\s)(?:day|week|month|year)s?(?:'s)?", re.I)
PLACE = re.compile(r"\b(?:laws of|courts of|jurisdiction of)\s+(?:the\s+)?(?:State of\s+)?[A-Z][a-zA-Z]+(?:,?\s+[A-Z][a-zA-Z]+)?")
PARTY = re.compile(r"\b[A-Z][\w&'’-]*(?:\s+[A-Z][\w&'’-]*){0,4}\s+(?:Pvt\.?\s+Ltd\.?|Private Limited|Limited|Ltd\.?|LLP|LLC|Inc\.?|Corp\.?|Corporation|Company)")
DEFINED = re.compile(r"\(the\s+[“\"]([^”\"]+)[”\"]\)")
REGEX_ENTITIES = [("date", DATE), ("money", MONEY), ("term", TERM), ("place", PLACE), ("party", PARTY)]
SPACY_MAP = {"ORG": "party", "PERSON": "party", "GPE": "place", "MONEY": "money"}

RISKS = [
    (r"unlimited", "Unlimited liability", 3, "No cap on how much one side may owe."),
    (r"without (?:prior )?notice", "Termination without notice", 3, "The agreement can end suddenly."),
    (r"sole discretion", "Sole discretion", 2, "One party decides alone, with no objective test."),
    (r"automatically renew|auto-?renew", "Auto-renewal", 2, "You stay bound unless you cancel in time."),
    (r"non-?compete|shall not provide similar services", "Non-compete", 2, "Limits who else you can work with."),
    (r"indemnif", "Indemnity", 2, "You may have to cover the other side's losses."),
    (r"penalty|liquidated damages", "Penalty", 2, "Extra charges apply for lateness or breach."),
    (r"irrevocabl|perpetual", "Irrevocable or perpetual right", 2, "Hard to undo once granted."),
    (r"waive", "Waiver of rights", 2, "A legal right is given up."),
    (r"as is|without warranty", "No warranty", 1, "No promise about quality."),
    (r"exclusive jurisdiction", "Exclusive jurisdiction", 1, "Disputes must be handled in one court."),
]
RISKS = [(re.compile(p, re.I), *rest) for p, *rest in RISKS]
ABBR = re.compile(r"(\b(?:Ltd|Pvt|Inc|Corp|Co|No|Mr|Ms|Dr|Rs|vs)\.|^\d+\.)$")
OBLIGATION = re.compile(r"\b(shall not|must not|shall|must|agrees? to|is required to|undertakes?|will)\b", re.I)

_summarizer = None


def split_sentences(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parts = nltk.sent_tokenize(line)
        except LookupError:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z“\"(0-9])", line)
        buf = ""
        for p in parts:
            buf = f"{buf} {p}".strip()
            if not ABBR.search(buf):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    return out


class ClauseClassifier:
    """TF-IDF + Logistic Regression trained on data/legal_clauses.csv (Pandas)."""

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = self.train()

    @staticmethod
    def build():
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english"),
            LogisticRegression(max_iter=2000, C=10))

    def train(self):
        df = pd.read_csv(DATA_CSV)
        model = self.build()
        model.fit(df["text"], df["label"])
        return model

    def predict(self, sentences):
        if not sentences:
            return []
        probs = self.model.predict_proba(sentences)
        idx = probs.argmax(axis=1)
        return [(self.model.classes_[i], float(p[i])) for p, i in zip(probs, idx)]


class LegalAnalyzer:
    def __init__(self, use_spacy=True):
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None  # falls back to regex entities only
        self.clf = ClauseClassifier()

    def analyze(self, text):
        text = text.replace("\r\n", "\n").strip()
        sents = split_sentences(text)
        spans = self._entities(text)
        ents = {}
        for sp in spans:
            if sp["type"] != "risk":
                key = re.sub(r"\s+", " ", sp["v"])
                bucket = ents.setdefault(sp["type"], {})
                bucket[key] = bucket.get(key, 0) + 1
        parties = list(ents.get("party", {}))

        long_sents = [(i, s) for i, s in enumerate(sents) if len(s.split()) >= 5]
        preds = self.clf.predict([s for _, s in long_sents])
        clauses = [{"k": lab, "s": s, "i": i, "conf": round(p, 2)}
                   for (i, s), (lab, p) in zip(long_sents, preds) if p >= 0.2]

        obligations = []
        for s in sents:
            m = OBLIGATION.search(s)
            if not m:
                continue
            due = TERM.search(s) or DATE.search(s)
            obligations.append({"s": s,
                                "actor": next((p for p in parties if p in s), "Not stated"),
                                "ban": "not" in m.group(0).lower(),
                                "due": due.group(0) if due else ""})

        risks = [{"s": s, "label": lab, "sev": sev, "why": why}
                 for s in sents for rx, lab, sev, why in RISKS if rx.search(s)]
        weight = {3: 18, 2: 10, 1: 4}
        score = min(100, sum(weight[r["sev"]] for r in risks))
        summary = self._abstractive(text) + self._extractive(sents, {c["s"] for c in clauses})
        return {"text": text, "sents": sents, "spans": spans, "ents": ents, "clauses": clauses,
                "obl": obligations, "risks": risks, "score": score, "summary": summary}

    def _entities(self, text):
        spans = []
        for typ, rx in REGEX_ENTITIES:
            spans += [{"s": m.start(), "e": m.end(), "type": typ, "v": m.group(0).strip()} for m in rx.finditer(text)]
        for m in DEFINED.finditer(text):
            spans.append({"s": m.start(1), "e": m.end(1), "type": "party", "v": m.group(1)})
        if self.nlp:
            defined = {m.group(1) for m in DEFINED.finditer(text)}
            for e in self.nlp(text[:200000]).ents:
                typ = SPACY_MAP.get(e.label_)
                if typ and self._keep_spacy(text, e, typ, defined):
                    spans.append({"s": e.start_char, "e": e.end_char, "type": typ, "v": e.text})
        for rx, lab, *_ in RISKS:
            spans += [{"s": m.start(), "e": m.end(), "type": "risk", "v": m.group(0)} for m in rx.finditer(text)]
        spans.sort(key=lambda x: (x["s"], -(x["e"] - x["s"])))
        kept, last = [], 0
        for sp in spans:
            if sp["s"] >= last:
                kept.append(sp)
                last = sp["e"]
        return kept

    @staticmethod
    def _keep_spacy(text, e, typ, defined):
        """Drop common spaCy mistakes on contracts (headings, defined terms, boilerplate nouns)."""
        t = e.text.strip()
        if '"' in t or "“" in t or t in defined or t.lower().startswith("the "):
            return False
        if re.search(r"Confidential|Information|Schedule|Agreement|Property|Compete|Services|Liability|Termination", t):
            return False
        if re.search(r"\d+\.\s*$", text[max(0, e.start_char - 6):e.start_char]):
            return False  # clause heading such as '5. Liability.'
        if typ == "party" and len(t.split()) < 2:
            return False
        return True

    @staticmethod
    def _extractive(sents, clause_set, k=4):
        cand = [(i, s) for i, s in enumerate(sents) if len(s.split()) >= 6]
        if not cand:
            return []
        X = TfidfVectorizer(stop_words="english").fit_transform([s for _, s in cand])
        scores = X.sum(axis=1).A1
        ranked = []
        for (i, s), sc in zip(cand, scores):
            sc = sc / (len(s.split()) ** 0.5)
            if s in clause_set:
                sc *= 1.3
            if re.search(r"\d", s):
                sc *= 1.15
            ranked.append((sc, i, s))
        top = sorted(sorted(ranked, reverse=True)[:k], key=lambda x: x[1])
        return [s for _, _, s in top]

    @staticmethod
    def _abstractive(text):
        """Hugging Face summary; only runs when USE_TRANSFORMERS=1."""
        global _summarizer
        if os.getenv("USE_TRANSFORMERS") != "1":
            return []
        try:
            from transformers import pipeline
            if _summarizer is None:
                _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            chunk = " ".join(text.split()[:700])
            return [_summarizer(chunk, max_length=90, min_length=30, do_sample=False)[0]["summary_text"]]
        except Exception:
            return []
