FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install --upgrade pip
RUN pip install pandas numpy scikit-learn pyarrow

COPY homework_4.py ./

ENTRYPOINT ["python", "homework_4.py", "--year", "2023", "--month", "05"]