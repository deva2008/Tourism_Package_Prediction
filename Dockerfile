
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app_streamlit /app/app_streamlit

# env variables used by the app for Hugging Face model id
ENV HF_USERNAME=""
ENV HF_MODEL_REPO=""

EXPOSE 7860
CMD ["streamlit", "run", "app_streamlit/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
