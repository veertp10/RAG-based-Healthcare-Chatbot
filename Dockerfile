FROM python:3.10

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ExtraTrees /app/ExtraTrees
COPY data/symptom_Description.csv /app/data/symptom_Description.csv
COPY data/symptom_precaution.csv /app/data/symptom_precaution.csv

WORKDIR /app
COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]