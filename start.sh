
echo "Configurando NLTK..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('rslp', quiet=True)"


echo "Iniciando Uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT