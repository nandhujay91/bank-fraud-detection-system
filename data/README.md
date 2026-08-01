## Files
- train_transaction.csv (590,540 rows, 394 cols) - main transaction data
- train_identity.csv (144,233 rows, 41 cols) - device/identity data 
  for subset of transactions, merged via TransactionID (left join)
- test_transaction.csv / test_identity.csv - same structure, no isFraud 
  label (Kaggle holdout set, not used for offline validation)
- sample_submission.csv - Kaggle output format template

## Target
isFraud: 0/1, 3.5% positive class (severe imbalance)

## Key columns
- TransactionDT: seconds from a fixed reference point (not a real 
  timestamp) - used for time-based split
- TransactionAmt: transaction amount
- ProductCD, card1-6, addr1-2: transaction/card metadata
- C1-C14, D1-D15: engineered features from Kaggle (counts, time deltas)
- M1-M9: match flags (categorical)
- id_01 to id_38, DeviceType, DeviceInfo: identity/device data, 
  high missingness (most transactions have no identity match)

## Data split
Time-based, 80/20. Train: TransactionDT 86,400-12,192,842 (~141 days). 
Test: 12,192,900-15,811,131 (~42 days). Fraud rate stable across split 
(3.51% train vs 3.44% test).