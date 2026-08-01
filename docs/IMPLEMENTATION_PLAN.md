## Week 1: Data Loading & EDA
- [x] Load train_transaction + train_identity
- [x] Merge on TransactionID
- [x] Confirm class imbalance (3.5% fraud)
- [x] Check missing data patterns
- [x] Implement time-based split (src/data_loader.py)

## Week 2: Feature Engineering
- [x] Velocity features (card1_txn_count_1d/7d, card1_txn_amt_sum_1d/7d)
- [x] Amount deviation features (expanding mean/std, no leakage)
- [x] Entity aggregation features (P_emaildomain_fraud_rate)
- [x] Entity aggregation features (card1_fraud_rate)
- [x] Categorical encoding (31 columns converted to category dtype)
- [ ] Apply full feature pipeline to test_df
- [ ] Save processed train/test data to data/processed/

## Week 3: Modeling
- [ ] Baseline model (Logistic Regression, class-weighted)
- [ ] LightGBM model
- [ ] MLflow experiment tracking
- [ ] Evaluation: PR-AUC, precision@k, cost-based threshold selection

## Week 4: Explainability
- [ ] SHAP values for model predictions
- [ ] Model card documentation

## Week 5: API & Deployment
- [ ] FastAPI /score endpoint
- [ ] Redis feature store integration
- [ ] Dockerize API

## Week 6: Monitoring & Testing
- [ ] Load testing (Locust)
- [ ] Drift monitoring (PSI/Evidently)
- [ ] Model decay simulation

## Week 7 (optional): Streaming
- [ ] Kafka producer/consumer simulation

## Week 8: Documentation & Packaging
- [ ] Final README
- [ ] Model card complete
- [ ] Repo cleanup