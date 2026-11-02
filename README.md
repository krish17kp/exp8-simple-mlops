# Experiment 8 — Simple MLOps Project

This project demonstrates the complete MLOps flow on a small delivery-time prediction model. The training examples are generated in code, so the project does not depend on Iris or an external dataset.

The model predicts delivery time from distance, package weight, and priority. The same model is available through Streamlit and a small Flask metrics service.

## Local run

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## MLOps components

- Git repository and GitHub Actions CI
- Docker image and Docker Compose stack
- Kubernetes Deployment and Services
- Prometheus metrics and Grafana dashboard configuration
- Streamlit Community Cloud entrypoint: `app.py`
- MLflow tracking under `mlruns/` (installed from `requirements-mlops.txt` for CI/Docker; Streamlit Cloud only needs `requirements.txt`)
