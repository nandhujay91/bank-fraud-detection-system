## Data Splitting Strategy

Dataset spans ~183 days (TransactionDT: 86,400 to 15,811,131 seconds).

Decision: Time-based split (80% earliest transactions -> train, 
20% most recent -> test). Random split rejected because:
- Fraud patterns evolve over time; random split leaks future 
  fraud tactics into training
- Production deployment only ever has past data to predict 
  future transactions - the offline split must mirror this

Train/test fraud rate comparison logged in 01_eda.ipynb to check 
for fraud-rate drift across the time period.