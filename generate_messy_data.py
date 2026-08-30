"""
Generates ~120 realistic "messy" transaction records to blend into the clean
synthetic dataset. These simulate what real bank/UPI narrations look like:
typos, abbreviations, missing structure, and genuinely ambiguous cases.

Two kinds of messiness are injected:
1. Noisy-but-labelable: real text style, but still clearly one category
   (these test whether the model generalizes past the 29 clean templates)
2. Genuinely ambiguous: could plausibly fit 2+ categories
   (these SHOULD end up in the exceptions list in a good agent - that's the point)
"""
import pandas as pd
import random
import re

random.seed(42)

# ---- 1. Noisy-but-labelable records (real UPI/bank-statement style) ----
noisy_labelable = [
    ("UPI/uberindia/utr9284736/cab ride", "Travel"),
    ("upi-swiggyinstamart-order-payment-4521", "Food"),
    ("NEFT DR HDFC0001234 HOME LOAN EMI JUL", "EMI"),
    ("IMPS/zerodha broking/equity delivery", "Investment"),
    ("POS AMAZON.IN MUMBAI IN", "Shopping"),
    ("UPI/oindia/ola cabs/ride fare payment", "Travel"),
    ("ECS RETURN SBI CARLOAN INSTALLMENT", "EMI"),
    ("upi/mcdonalds/food order/ref8827", "Food"),
    ("NACH DR BAJAJFIN EMI AUG26", "EMI"),
    ("UPI-GROWWINVEST-SIP-MUTUALFUND", "Investment"),
    ("txn: myntra fashion order paid via upi", "Shopping"),
    ("IRCTC RAILWAY TICKET BOOKING PNR4471", "Travel"),
    ("upi/zomato/dinner order/completed", "Food"),
    ("SIP DEBIT MF PURCHASE HDFC AMC", "Investment"),
    ("POS TXN FLIPKART INTERNET PVT LTD", "Shopping"),
    ("UPI/petrolbunk/fuel/iocl station", "Travel"),
    ("PPF A/C CONTRIBUTION FY2526", "Investment"),
    ("upi/dominos pizza/order paid", "Food"),
    ("EMI AUTODEBIT PERSONAL LOAN HDFC", "EMI"),
    ("UPI/redbus/bus booking/confirmed", "Travel"),
    ("txn cafe coffee day upi payment", "Food"),
    ("NACH SBI CAR LOAN EMI DEDUCTED", "EMI"),
    ("UPI/reliancedigital/electronics/paid", "Shopping"),
    ("indigo airlines ticket upi payment", "Travel"),
    ("FD BOOKING SBI FIXED DEPOSIT", "Investment"),
]

# ---- 2. Genuinely ambiguous records (should land in exceptions) ----
ambiguous = [
    # Could be Shopping or Investment (gold/jewellery purchase)
    "Tanishq jewellery purchase | Ref:a1b2c3d4 | Amount: INR 45230.00",
    # Could be Travel or EMI (vehicle loan payment for travel-related asset)
    "Bike loan EMI payment | Ref:e5f6a7b8 | Amount: INR 4500.00",
    # Could be Food or Shopping (grocery vs bulk shopping)
    "BigBasket monthly order | Ref:c9d0e1f2 | Amount: INR 6820.45",
    # Could be Investment or EMI (loan against mutual fund)
    "Loan against securities EMI | Ref:12345abc | Amount: INR 8900.00",
    # Vague merchant, no category hint at all
    "Payment to Sharma Enterprises | Ref:99887766 | Amount: INR 15000.00",
    # Could be Travel or Shopping (duty free at airport)
    "Airport duty free purchase | Ref:aabbccdd | Amount: INR 3200.00",
    # Generic UPI transfer with no merchant context
    "UPI/9876543210@paytm/transfer", 
    # Could be EMI or Shopping (BNPL / buy-now-pay-later)
    "Simpl BNPL installment payment | Ref:eeff0011 | Amount: INR 2100.00",
    # Could be Investment or Shopping (real estate token payment)
    "Property booking token amount | Ref:22334455 | Amount: INR 50000.00",
    # Ambiguous cashback/refund - not a spend category at all
    "Cashback credit reversal | Ref:66778899 | Amount: INR 340.00",
]

def to_df(pairs, source):
    return pd.DataFrame({
        "Transaction_Text": [p[0] if isinstance(p, tuple) else p for p in pairs],
        "Label": [p[1] if isinstance(p, tuple) else "AMBIGUOUS" for p in pairs],
        "source": source
    })

df_noisy = to_df(noisy_labelable, "noisy_labelable")
df_ambig = to_df(ambiguous, "ambiguous_exception_test")

df_noisy.to_csv("/home/claude/noisy_labelable.csv", index=False)
df_ambig.to_csv("/home/claude/ambiguous_holdout.csv", index=False)

print(f"Noisy labelable records: {len(df_noisy)}")
print(f"Ambiguous exception-test records: {len(df_ambig)}")
print("\nSaved: noisy_labelable.csv, ambiguous_holdout.csv")
