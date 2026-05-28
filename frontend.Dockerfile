FROM python:3.13-alpine3.23

WORKDIR /frontend

RUN pip install --no-cache-dir streamlit

COPY frontend .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
