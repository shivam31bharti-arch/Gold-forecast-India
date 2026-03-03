# CI/CD

Pipeline stages:
1. lint & unit tests
2. data schema validation
3. model training (smoke) on small dataset
4. integration tests (inference on sample)
5. build Docker image
6. deploy to staging

Use GitHub Actions or GitLab CI.
