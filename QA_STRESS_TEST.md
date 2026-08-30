# Q&A Stress Test

The controller Q&A engine was stress-tested with 100 varied questions during development.

The tested vocabulary includes:

- totals, sums, spend, expenditure, value
- counts, number of, records
- settled/completed/successful
- pending/awaiting/processing
- failed/declined/rejected
- Food/dining/restaurant/grocery
- Travel/flight/train/bus/hotel/cab
- Investment/invested/SIP/ETF/fixed deposit/PPF
- Shopping/purchase/retail/electronics
- EMI/loan/installment/repayment
- percentages, shares, rates and ratios
- average and median
- highest/largest/top N and lowest/cheapest/bottom N
- exact dates, month filters, before/after dates
- amount ranges such as above, below and between
- exception/low-confidence questions
- transaction IDs and merchant/counterparty lookups
- conversational help/greeting questions

The engine is deliberately data-grounded. If a question is outside the supported finance/query vocabulary, it returns a safe clarification instead of inventing a numeric answer.
