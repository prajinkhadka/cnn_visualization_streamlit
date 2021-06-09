FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
WORKDIR /app
ENTRYPOINT ["streamlit", "run"]
CMD ["run.py"]
