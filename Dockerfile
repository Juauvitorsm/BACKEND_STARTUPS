FROM python:3.11-slim

ENV NLTK_DATA=/app/nltk_data
ENV PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p ${NLTK_DATA} \
    && python -c "import nltk; \
                  nltk.download('punkt', download_dir='${NLTK_DATA}', quiet=True); \
                  nltk.download('stopwords', download_dir='${NLTK_DATA}', quiet=True); \
                  nltk.download('rslp', download_dir='${NLTK_DATA}', quiet=True)"

COPY . .

CMD ["python", "backend/app/main.py"]