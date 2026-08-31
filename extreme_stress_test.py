from pathlib import Path
import joblib,pandas as pd
from sklearn.metrics import accuracy_score,classification_report
from classifier_utils import clean_transaction_text
BASE=Path(__file__).resolve().parent
model=joblib.load(BASE/'finance_classifier_model.joblib')
vectorizer=joblib.load(BASE/'finance_vectorizer.joblib')
df=pd.read_csv(BASE/'extreme_unseen_holdout.csv')
X=vectorizer.transform(df.Transaction_Text.map(clean_transaction_text))
df['Predicted']=model.predict(X)
df['Confidence']=model.predict_proba(X).max(axis=1)
df['Exception']=df.Confidence<.60
print(f'EXTREME HOLDOUT CASES: {len(df)}')
print(f'ACCURACY: {accuracy_score(df.Expected,df.Predicted):.2%}')
print(classification_report(df.Expected,df.Predicted))
wrong=df[df.Expected!=df.Predicted]
print('\nMISCLASSIFIED:')
print(wrong.to_string(index=False) if len(wrong) else 'None')
print(f'\nLOW-CONFIDENCE CASES: {int(df.Exception.sum())}')
print(f'LOW-CONFIDENCE CORRECT: {int(((df.Expected==df.Predicted)&df.Exception).sum())}')
print(f'LOW-CONFIDENCE WRONG: {int(((df.Expected!=df.Predicted)&df.Exception).sum())}')
df.to_csv(BASE/'extreme_stress_results.csv',index=False)
