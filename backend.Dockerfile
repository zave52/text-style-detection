FROM python:3.13-slim-trixie

WORKDIR /app

RUN pip install --no-cache-dir fastapi joblib uvicorn scikit-learn spacy

COPY app ./app
COPY saving ./saving

EXPOSE 8000

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
