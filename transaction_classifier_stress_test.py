from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from classifier_utils import clean_transaction_text

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / 'finance_classifier_model.joblib')
vectorizer = joblib.load(BASE / 'finance_vectorizer.joblib')

cases = {
    'Food': [
        'UPI/SWIGGY/MEAL/8821','upi-zomato-dinner-paid','POS KFC HYDERABAD','MCD meal txn',
        'DOMINOS PIZZA ORDER','PIZZA HUT TAKEAWAY','STARBUCKS CAFE BILL','CCD COFFEE PAYMENT',
        'restaurant qr payment','food delivery completed','grocery delivery instamart','BIGBASKET MONTHLY ORDER',
        'ZEpto groceries','blinkit food order','canteen lunch payment','bakery purchase','dinner at restaurant',
        'meal expense paid by card','food court transaction','supermarket grocery bill'
    ],
    'Travel': [
        'UPI/UBERINDIA/UTR9284736/CAB RIDE','Ola ride fare','RAPIDO BIKE TAXI PAYMENT','POS INDIGO AIRLINES',
        'INDIGO FLIGHT TICKET','AIR INDIA AIRFARE','AKASA AIR TICKET','IRCTC PNR4471',
        'railway reservation','redbus bus booking','MMT HOTEL STAY','MAKEMYTRIP RESERVATION',
        'OYO HOTEL PAYMENT','airport cab fare','metro travel recharge','IOCL FUEL STATION','petrol pump payment',
        'taxi fare payment','travel booking confirmed','hotel accommodation charge'
    ],
    'EMI': [
        'NACH DR HOMELOAN HDFC','home loan repayment debit','SBI CAR LOAN EMI DEDUCTED','bike finance monthly payment',
        'two wheeler loan installment','CC EMI DEBIT','credit card installment','PERSONAL LOAN AUTODEBIT',
        'BAJAJ FINANCE INSTALLMENT','consumer durable installment','loan payment due','mortgage installment',
        'education loan repayment','BNPL repayment','SIMPL INSTALLMENT','paylater monthly installment',
        'NACH loan debit','ECS loan installment','vehicle finance repayment','monthly loan deduction'
    ],
    'Investment': [
        'IMPS/ZERODHA BROKING/EQUITY DELIVERY','UPI GROWW SIP','MF SIP AUTO DEBIT','mutual fund purchase',
        'FD BOOKING','fixed deposit investment','DEMAT STOCK BUY','UPSTOX EQUITY PURCHASE','ANGEL ONE SHARES',
        'ETF INVESTMENT','NPS CONTRIBUTION','PPF A/C CONTRIBUTION','bond subscription','government bond purchase',
        'stock market investment','equity delivery debit','portfolio investment','retirement contribution',
        'recurring deposit investment','securities purchase'
    ],
    'Shopping': [
        'POS AMAZON.IN MUMBAI IN','AMAZON ONLINE ORDER','POS CROMA ELECTRONICS','FLIPKART ORDER ID',
        'MYNTRA FASHION ORDER','AJIO CLOTHING PURCHASE','MEESHO ORDER','RELIANCE DIGITAL ELECTRONICS',
        'TITAN RETAIL PURCHASE','TANISHQ JEWELLERY','IKEA HOME FURNISHING','DECATHLON SHOPPING',
        'NYKAA COSMETICS ORDER','retail store card payment','electronics purchase','clothing purchase',
        'home appliance purchase','mobile phone purchase','laptop purchase','online ecommerce order'
    ]
}
rows=[]
for label, texts in cases.items():
    for text in texts: rows.append((text,label))
df=pd.DataFrame(rows, columns=['Transaction_Text','Expected'])
X=vectorizer.transform(df.Transaction_Text.map(clean_transaction_text))
df['Predicted']=model.predict(X)
df['Confidence']=model.predict_proba(X).max(axis=1)
df['Exception']=df.Confidence < .60

acc=accuracy_score(df.Expected, df.Predicted)
print(f'STRESS CASES: {len(df)}')
print(f'ACCURACY: {acc:.2%}')
print(classification_report(df.Expected, df.Predicted))
wrong=df[df.Expected != df.Predicted]
print('\nMISCLASSIFIED:')
print(wrong.to_string(index=False) if len(wrong) else 'None')
print('\nLOW-CONFIDENCE BUT CORRECT (these should remain reviewable):')
print(df[(df.Expected==df.Predicted) & df.Exception].to_string(index=False))
print('\nAll categories covered:', ', '.join(sorted(df.Expected.unique())))
df.to_csv(BASE/'transaction_classifier_stress_results.csv', index=False)
