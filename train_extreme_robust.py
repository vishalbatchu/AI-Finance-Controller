from pathlib import Path
import joblib,pandas as pd,numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from classifier_utils import clean_transaction_text
BASE=Path(__file__).resolve().parent
base=pd.read_csv(BASE/'trainable_dataset.csv')[['Transaction_Text','Label']]
noisy=pd.read_csv(BASE/'noisy_labelable.csv')[['Transaction_Text','Label']]
hard=pd.read_csv(BASE/'extreme_hard_train.csv')[['Transaction_Text','Label']]
# Keep the existing curated paraphrases and hard examples.
aug={
'Food':['swiggy food order','zomato restaurant payment','mcdonalds meal','kfc dinner','dominos takeaway','blinkit grocery order','bigbasket groceries','restaurant card purchase','upi food delivery','dinner bill paid','food order payment','meal expense'],
'Travel':['uber cab ride','ola ride fare','rapido bike taxi','irctc rail ticket','redbus ticket','indigo flight ticket','air india airfare','makemytrip hotel','oyo hotel stay','airport transfer','taxi fare','fuel station payment','flight booking payment'],
'EMI':['home loan installment','car loan installment','bike loan installment','credit card emi','bajaj finserv installment','loan auto debit','monthly loan installment','mortgage payment','education loan installment','nach loan installment','bnpl installment','paylater installment','loan repayment debit'],
'Investment':['zerodha equity purchase','groww mutual fund sip','fd booking','mutual fund investment','sip debit','shares purchase','equity delivery','etf purchase','demat investment','nps contribution','ppf contribution','bonds purchase','stock market investment'],
'Shopping':['amazon online order','flipkart purchase','myntra clothing order','ajio fashion purchase','meesho order','croma electronics purchase','titan purchase','tanishq jewellery purchase','ikea furniture purchase','decathlon shopping','electronics purchase','clothing purchase','online shopping payment']}
augdf=pd.DataFrame([(x,c) for c,v in aug.items() for x in v],columns=['Transaction_Text','Label'])
targeted = pd.DataFrame({
 'Transaction_Text': [
  'supermarket dinner order','supermarket lunch order','grocery dinner payment','food supermarket order',
  'eqiuty purchase','equityy purchase','equtiy delivery','sharse purchase','sharees purchase','shars purchase','shares buy','stock buy',
  'NEFT shares purchase','CARD equity purchase','UPI equity delivery','demat shares purchase','mutualfund sip debit',
  'AMAZON product purchase','FLIPKART product purchase','retail product purchase'
 ],
 'Label': ['Food','Food','Food','Food','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Investment','Shopping','Shopping','Shopping']
})
augdf=pd.concat([augdf,targeted],ignore_index=True)
all_df=pd.concat([base,noisy,augdf,hard],ignore_index=True).drop_duplicates('Transaction_Text').dropna()
all_df['clean_text']=all_df.Transaction_Text.map(clean_transaction_text)
vectorizer=FeatureUnion([
 ('word',TfidfVectorizer(ngram_range=(1,3),min_df=1,max_features=18000,sublinear_tf=True,strip_accents='unicode')),
 ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(2,6),min_df=1,max_features=30000,sublinear_tf=True,strip_accents='unicode')),
])
X=vectorizer.fit_transform(all_df.clean_text)
model=LogisticRegression(max_iter=3000,C=2.0,class_weight='balanced')
model.fit(X,all_df.Label)
joblib.dump(model,BASE/'finance_classifier_model.joblib'); joblib.dump(vectorizer,BASE/'finance_vectorizer.joblib')
# Extreme unseen evaluation.
h=pd.read_csv(BASE/'extreme_unseen_holdout.csv')
Xh=vectorizer.transform(h.Transaction_Text.map(clean_transaction_text)); p=model.predict(Xh); probs=model.predict_proba(Xh).max(axis=1)
h['Predicted']=p; h['Confidence']=probs; h['Exception']=probs<.60
h.to_csv(BASE/'extreme_stress_results.csv',index=False)
print('TRAINING ROWS:',len(all_df))
print('EXTREME HOLDOUT:',len(h))
print('ACCURACY:',accuracy_score(h.Expected,p))
print(classification_report(h.Expected,p))
wrong=h[h.Expected!=h.Predicted]
print('MISCLASSIFIED:',len(wrong)); print(wrong.head(30).to_string(index=False))
print('LOW CONFIDENCE:',int((probs<.60).sum()))
# Save metadata.
meta={'training_rows':int(len(all_df)),'extreme_holdout_rows':int(len(h)),'extreme_holdout_accuracy':float(accuracy_score(h.Expected,p)),'low_confidence_extreme_cases':int((probs<.60).sum()),'feature_union':'word 1-3 + char_wb 2-6','confidence_threshold':0.60,'trained_mode':'extreme robust augmentation'}
import json
(BASE/'extreme_model_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
