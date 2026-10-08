"""Train and evaluate the clause classifier:  python train_classifier.py"""
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report
from analyzer.pipeline import ClauseClassifier, DATA_CSV, MODEL_PATH

df = pd.read_csv(DATA_CSV)
print(df["label"].value_counts().to_string(), "\n")
pred = cross_val_predict(ClauseClassifier.build(), df["text"], df["label"],
                         cv=StratifiedKFold(4, shuffle=True, random_state=42))
print(classification_report(df["label"], pred, zero_division=0))
model = ClauseClassifier.build().fit(df["text"], df["label"])
joblib.dump(model, MODEL_PATH)
print("Saved", MODEL_PATH)
