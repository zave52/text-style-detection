FROM python:3.13-alpine3.23

WORKDIR /app

RUN apk add --no-cache libgomp libstdc++ \
	&& apk add --no-cache --virtual build-deps \
	g++ \
    && pip install --no-cache-dir fastapi joblib uvicorn scikit-learn spacy \
    && apk del build-deps

COPY app .
COPY saving ./saving

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
