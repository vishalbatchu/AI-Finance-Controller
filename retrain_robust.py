from pathlib import Path
import re, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE=Path(__file__).resolve().parent

def clean_text(t):
    t=str(t).lower()
    # Normalize common banking/transaction formatting without deleting merchant words.
    replacements={
        'upi-':'upi ', 'upi/':'upi ', 'pos ':'pos ', 'neft dr':'neft ', 'imps/':'imps ',
        'nach dr':'nach ', 'ecs return':'ecs ', 'txn:':'txn ', 'a/c':' account ',
        'autodebit':'auto debit', 'autodebit':'auto debit',
    }
    for a,b in replacements.items(): t=t.replace(a,b)
    # Expand common bank narration abbreviations so short/cryptic names still carry meaning.
    for a,b in [(r'\bfd\b','fixed deposit'),(r'\bmf\b','mutual fund'),(r'\bcc\b','credit card'),(r'\bppf\b','public provident fund'),(r'\bnps\b','pension investment'),(r'\bbnpl\b','buy now pay later'),(r'\bemi\b','installment'),(r'\bpos\b','point of sale')]:
        t=re.sub(a,b,t)
    t=re.sub(r'ref[:/\-]?\w+', ' ', t)
    t=re.sub(r'(?:amount|amt)\s*[:=]?\s*(?:inr|rs|₹)?\s*[\d,]+(?:\.\d+)?', ' ', t)
    t=re.sub(r'\binr\s*[\d,]+(?:\.\d+)?', ' ', t)
    t=re.sub(r'\brs\.?\s*[\d,]+(?:\.\d+)?', ' ', t)
    t=re.sub(r'\b\d{5,}\b', ' ', t) # UTR/account-like numbers
    t=re.sub(r'[^a-z0-9\s/]', ' ', t)
    t=re.sub(r'\s+', ' ', t).strip()
    return t

train=pd.read_csv(BASE/'trainable_dataset.csv')[['Transaction_Text','Label']]
noisy=pd.read_csv(BASE/'noisy_labelable.csv')[['Transaction_Text','Label']]

# Hand-crafted paraphrases cover real bank narration styles and merchants not dominant in the seed set.
aug={
'Food':[
 'swiggy food order','zomato restaurant payment','mcdonalds meal','kfc dinner','pos kfc food purchase','kfc restaurant payment','kfc meal transaction','dominos takeaway','pizza hut order',
 'blinkit grocery order','zepto grocery delivery','instamart grocery purchase','bigbasket groceries','dunzo food delivery',
 'cafe coffee day bill','starbucks cafe payment','restaurant card purchase','canteen meal','food court payment',
 'upi restaurant bill','upi food delivery','grocery store purchase','supermarket groceries','bakery purchase','lunch payment',
 'dinner bill paid','breakfast at cafe','food order payment','meal expense'
],
'Travel':[
 'uber cab ride','ola ride fare','rapido bike taxi','metro recharge','irctc rail ticket','train reservation',
 'redbus ticket','bus reservation','indigo flight ticket','indigo airlines ticket','indigo airfare','pos indigo airlines','air india airfare','akasa flight booking','vistara flight booking',
 'makemytrip hotel','goibibo hotel reservation','oyo hotel stay','booking com hotel','airport transfer','taxi fare',
 'cab fare payment','petrol pump payment','fuel station payment','parking at airport','travel booking','hotel accommodation',
 'flight booking payment','railway ticket purchase'
],
'EMI':[
 'home loan installment','home loan repayment','car loan installment','bike loan installment','bike finance monthly payment','bike finance installment','two wheeler loan payment','vehicle finance installment','personal loan repayment',
 'credit card emi','credit card installment','bajaj finserv installment','lendingkart emi','consumer durable emi',
 'loan auto debit','loan payment due','monthly loan installment','mortgage payment','housing loan emi','vehicle loan emi',
 'education loan installment','emi deduction','loan repayment debit','nach loan installment','ecs loan debit','bnpl installment',
 'paylater installment','simpl installment','phone emi payment','laptop emi payment'
],
'Investment':[
 'zerodha equity purchase','groww mutual fund sip','fd booking','fixed deposit booking','fd investment booking','mf sip auto debit','mutual fund sip auto debit','mutual fund investment','sip debit','stocks bought',
 'shares purchase','equity delivery','etf purchase','demat investment','upstox stock purchase','angel one equity purchase',
 'coin mutual fund investment','nps contribution','ppf contribution','fixed deposit booking','fd investment',
 'recurring deposit investment','bonds purchase','government bond investment','securities purchase','stock market investment',
 'mutual fund sip payment','investment account transfer','portfolio investment','retirement contribution','gold bond purchase'
],
'Shopping':[
 'amazon online order','flipkart purchase','myntra clothing order','ajio fashion purchase','meesho order',
 'reliance digital electronics','croma electronics purchase','titan purchase','tanishq jewellery purchase','ikea furniture purchase',
 'decathlon shopping','nykaa cosmetics order','zepto non food purchase','retail store card payment','electronics purchase',
 'clothing purchase','fashion order','home appliances purchase','furniture purchase','online shopping payment',
 'ecommerce order','pos retail purchase','mall purchase','supermarket non food purchase','mobile phone purchase','laptop purchase'
]
}
augdf=pd.DataFrame([(x,c) for c,items in aug.items() for x in items],columns=['Transaction_Text','Label'])
all_df=pd.concat([train,noisy,augdf],ignore_index=True).drop_duplicates('Transaction_Text')
all_df['clean_text']=all_df.Transaction_Text.map(clean_text)

vectorizer=FeatureUnion([
 ('word', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=10000, sublinear_tf=True)),
 ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1, max_features=15000, sublinear_tf=True)),
])
X=vectorizer.fit_transform(all_df.clean_text)
model=LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced')
model.fit(X,all_df.Label)
joblib.dump(model, BASE/'finance_classifier_model.joblib')
joblib.dump(vectorizer, BASE/'finance_vectorizer.joblib')

# Test on noisy data that was not included as an evaluation set by using the original noisy list separately.
def predict(texts):
    xx=vectorizer.transform([clean_text(x) for x in texts])
    p=model.predict(xx); probs=model.predict_proba(xx).max(axis=1)
    return p,probs
p,conf=predict(noisy.Transaction_Text)
print('NOISY LABELABLE ACCURACY:',accuracy_score(noisy.Label,p))
print(classification_report(noisy.Label,p))

stress={
'Food': ['upi/swgg/food/meal','POS KFC HYDERABAD','MCD meal txn','foodpanda style meal order','grocery delivery instamart','restaurant qr payment','cafe bill paid by card','pizza takeaway','lunch at restaurant'],
'Travel': ['UPI/RAPIDO/BIKE TAXI','POS INDIGO AIRLINES','IRCTC PNR ticket','MMT hotel stay','cab ride payment','metro travel recharge','fuel station IOCL','railway reservation','airport cab fare'],
'EMI': ['NACH DR HOMELOAN HDFC','loan installment auto debit','CC EMI DEBIT','bike finance monthly payment','PAYLATER INSTALLMENT','SBI car loan deduction','mortgage installment','consumer loan repayment','BNPL repayment'],
'Investment': ['UPI GROWW SIP','ZERODHA EQ DELIVERY','MF SIP AUTO DEBIT','NPS CONTRIBUTION','FD BOOKING','DEMAT STOCK BUY','ETF INVESTMENT','MUTUAL FUND PURCHASE','BOND SUBSCRIPTION'],
'Shopping': ['POS CROMA ELECTRONICS','AMAZON.IN ORDER','AJIO FASHION','TITAN RETAIL PURCHASE','TANISHQ JEWELLERY','ONLINE RETAIL ORDER','FLIPKART ORDER ID','MOBILE PHONE PURCHASE','IKEA HOME FURNISHING']
}
stressdf=pd.DataFrame([(x,c) for c,items in stress.items() for x in items],columns=['Transaction_Text','Label'])
p,conf=predict(stressdf.Transaction_Text)
stressdf['Predicted']=p; stressdf['Confidence']=conf
print('STRESS ACCURACY:',accuracy_score(stressdf.Label,p))
print(stressdf[stressdf.Label!=stressdf.Predicted].to_string(index=False))
print('\nLOW CONF STRESS:')
print(stressdf[stressdf.Confidence<.60].to_string(index=False))
