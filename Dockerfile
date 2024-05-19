FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY random_forest_model.joblib .

COPY scaler.joblib .

COPY flask_app.py .

EXPOSE 5000

CMD ["python", "flask_app.py"]
