from pathlib import Path
import random,re,pandas as pd
BASE=Path(__file__).resolve().parent
random.seed(20260830)

CATEGORIES={
'Food': ['swiggy','zomato','mcdonalds','kfc','dominos','pizzahut','starbucks','ccd','blinkit','zepto','instamart','bigbasket','dunzo','restaurant','cafe','bakery','canteen','foodcourt','grocery','supermarket'],
'Travel': ['uber','ola','rapido','irctc','redbus','indigo','airindia','akasa','makemytrip','goibibo','oyo','booking','airport','metro','petrol','iocl','railway','taxi','fuel'],
'EMI': ['homeloan','carloan','bikeloan','vehiclefinance','personalloan','creditcard','bajajfinance','lendingkart','mortgage','educationloan','consumerloan','nach','ecs','bnpl','paylater','simpl','installment','emi'],
'Investment': ['zerodha','groww','mutualfund','sip','fd','fixeddeposit','upstox','angelone','demat','etf','nps','ppf','bond','stocks','shares','equity','portfolio','securities','recurringdeposit'],
'Shopping': ['amazon','flipkart','myntra','ajio','meesho','reliancedigital','croma','titan','tanishq','ikea','decathlon','nykaa','retail','electronics','clothing','furniture','appliance','mobilephone','laptop','ecommerce']}

prefixes=['UPI','POS','NEFT','IMPS','NACH','ECS','TXN','BANK']
noise=['PAYMENT','DEBIT','DR','PAID','SUCCESS','COMPLETED','ONLINE','CARD','ACCOUNT','MERCHANT','REF','TXN','ORDER','CHARGE']
seps=['/','-','_','*',' ',':']

def typo(s):
    if len(s)<5:return s
    op=random.choice(['drop','swap','repeat','near'])
    i=random.randrange(1,len(s)-1)
    if op=='drop': return s[:i]+s[i+1:]
    if op=='swap': return s[:i]+s[i+1]+s[i]+s[i+2:]
    if op=='repeat': return s[:i]+s[i]+s[i:]
    near={'a':'s','s':'a','i':'o','o':'i','e':'w','w':'e','m':'n','n':'m'}
    return s[:i]+near.get(s[i],s[i])+s[i+1:]

def compact(s): return re.sub(r'[^a-z0-9]','',s.lower())

def make_variant(cat, hard=True):
    key=random.choice(CATEGORIES[cat])
    forms={
      'Food': [f'{key} food order',f'{key} restaurant payment',f'{key} meal bill',f'{key} grocery purchase',f'{key} cafe payment',f'{key} dinner order'],
      'Travel':[f'{key} cab fare',f'{key} flight ticket',f'{key} rail ticket',f'{key} hotel booking',f'{key} travel payment',f'{key} fuel payment'],
      'EMI':[f'{key} loan installment',f'{key} emi payment',f'{key} monthly repayment',f'{key} auto debit',f'{key} installment deduction',f'{key} loan payment'],
      'Investment':[f'{key} investment',f'{key} purchase',f'{key} sip contribution',f'{key} equity delivery',f'{key} account contribution',f'{key} booking'],
      'Shopping':[f'{key} online order',f'{key} retail purchase',f'{key} shopping payment',f'{key} product purchase',f'{key} card purchase',f'{key} order payment']}
    text=random.choice(forms[cat])
    if not hard:return text
    transforms=random.sample(range(8), random.randint(2,5))
    for t in transforms:
      if t==0: text=random.choice(prefixes)+' '+text
      elif t==1: text=text+random.choice(seps)+str(random.randint(100,99999999))
      elif t==2: text=text.replace(' ',random.choice(seps))
      elif t==3: text=random.choice([text.upper(),text.title(),''.join(c.upper() if random.random()<.35 else c for c in text)])
      elif t==4: text=typo(text)
      elif t==5: text=' '.join(random.sample(text.split(),len(text.split())))
      elif t==6: text=text+' '+random.choice(noise)
      elif t==7: text=random.choice(['Ref','UTR','ID'])+random.choice(seps)+str(random.randint(10**6,10**10))+' '+text
    return text

# Hard training set: broad variants, separate random seed from holdout.
train_rows=[]
for cat in CATEGORIES:
    for _ in range(220): train_rows.append((make_variant(cat,True),cat,'extreme_augmented'))
train_df=pd.DataFrame(train_rows,columns=['Transaction_Text','Label','source']).drop_duplicates('Transaction_Text')
train_df.to_csv(BASE/'extreme_hard_train.csv',index=False)

# Evaluation set: 500 cases with different seed and deliberately more composition changes.
random.seed(987654)
hold=[]
for cat in CATEGORIES:
    for _ in range(100): hold.append((make_variant(cat,True),cat,'extreme_unseen_holdout'))
hold_df=pd.DataFrame(hold,columns=['Transaction_Text','Expected','source']).drop_duplicates('Transaction_Text').reset_index(drop=True)
hold_df.to_csv(BASE/'extreme_unseen_holdout.csv',index=False)
print('extreme_hard_train:',len(train_df))
print('extreme_unseen_holdout:',len(hold_df))
print('categories:',hold_df.Expected.value_counts().to_dict())
