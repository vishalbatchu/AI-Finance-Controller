"""Stress-test the finance Q&A engine with varied natural-language phrasing."""
import pandas as pd
from settlement_qa import SettlementQA

QUESTIONS = [
    "What is the total amount?",
    "How much money did we spend?",
    "Total spend on food",
    "How much did dining cost?",
    "How much did we spend on travel?",
    "How much went to investments?",
    "Total shopping spend",
    "How much in EMIs?",
    "How many transactions are pending?",
    "How many completed payments?",
    "How many declined payments?",
    "What percentage is pending?",
    "What is the failure rate?",
    "What is the completion rate?",
    "What share is Food?",
    "What proportion is Travel?",
    "Average transaction value",
    "Median pending amount",
    "Largest transaction",
    "Top 5 travel expenses",
    "3 cheapest shopping transactions",
    "Failed travel transactions",
    "Pending shopping transactions",
    "Transactions after 2026-08-01",
    "Transactions before 2026-08-01",
    "Payments over INR 20000",
    "Payments under INR 5000",
    "Transactions between INR 10000 and INR 20000",
    "How much was spent in July?",
    "How much was spent in August?",
    "How many transactions need review?",
    "What is the match rate?",
    "Details for TXN100003",
    "What did we pay at Dominos?",
    "How much did IKEA cost?",
    "Amount paid to HDFC",
    "Show Amazon transactions",
    "What can you answer?",
]


def main():
    df = pd.read_csv("settlement_batch.csv")
    qa = SettlementQA(df)
    failures = []
    for question in QUESTIONS:
        answer = qa.answer(question)
        if answer.startswith("I couldn't confidently"):
            failures.append(question)
        print(f"Q: {question}\nA: {answer}\n")

    print("=" * 72)
    print(f"Questions tested: {len(QUESTIONS)}")
    print(f"Recognized: {len(QUESTIONS) - len(failures)}")
    print(f"Rejected/unsupported: {len(failures)}")
    if failures:
        print("Unsupported questions:")
        for item in failures:
            print(" -", item)


if __name__ == "__main__":
    main()
