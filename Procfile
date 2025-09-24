release: python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('rslp')"
web: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT