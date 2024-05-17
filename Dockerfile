FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Crop_Recommendation.csv .

COPY flask_app.py .

EXPOSE 5000

CMD ["python", "flask_app.py"]
