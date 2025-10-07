
mkdir -p /app/nltk_data


export NLTK_DATA=/app/nltk_data


echo "Forçando download de recursos NLTK para $NLTK_DATA"
python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA', quiet=True); nltk.download('stopwords', download_dir='$NLTK_DATA', quiet=True); nltk.download('rslp', download_dir='$NLTK_DATA', quiet=True)"


echo "Iniciando Uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port \$PORT