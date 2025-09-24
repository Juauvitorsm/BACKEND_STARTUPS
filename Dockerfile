# Usa uma imagem base do Python que já vem com as ferramentas de build necessárias
FROM python:3.13-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Copia os arquivos de dependência e os instala
COPY backend/requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copia o restante da aplicação
COPY backend /app/backend

# Define o comando de inicialização da sua API
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]