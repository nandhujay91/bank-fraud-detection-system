## Feature Set

Final engineered feature set: 441 features (443 total columns minus 
TransactionID and isFraud).

Breakdown:
- 434 raw columns from train_transaction + train_identity (merged)
- 9 engineered features added:
  - card1_txn_count_1d, card1_txn_count_7d (velocity: transaction 
    frequency per card)
  - card1_txn_amt_sum_1d, card1_txn_amt_sum_7d (velocity: spend volume 
    per card)
  - card1_amt_expanding_mean, card1_amt_expanding_std (running 
    statistics per card)
  - amt_deviation_from_avg (how unusual is this transaction's amount 
    vs the card's history)
  - P_emaildomain_fraud_rate, card1_fraud_rate (historical fraud rate 
    per entity, no leakage - only uses transactions before current one)

## Feature Selection Strategy (planned, post-baseline)

Training starts with the full 441-feature set to establish a baseline.
After initial LightGBM training, feature importance (SHAP + gain-based)
will be used to prune to a smaller, more production-viable feature set,
mirroring real banking practice:
- Regulatory interpretability favors fewer, explainable features
- Real-time features add latency/infra cost (Redis lookups) - low-value 
  features get cut to reduce production cost
- Correlated/redundant features get pruned after importance analysis
- Any feature with fairness/compliance risk (proxies for protected 
  characteristics) gets flagged for review

Target: reduce from 441 to a documented smaller set (~50-100 features) 
while tracking any PR-AUC change, and documenting the reasoning for 
each dropped feature in model_card.md.