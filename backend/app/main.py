import uvicorn
import nltk
import warnings 
warnings.filterwarnings(
    "ignore", 
    message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None", 
    category=UserWarning
)
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
from contextlib import asynccontextmanager

from .search_engine import SearchEngine
from .database import engine, Base, get_db
from . import models, security, schemas
from .schemas import UserLogin

search_engine_instance: Optional[SearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global search_engine_instance
    
    print("Verificando e baixando recursos do NLTK...")
    try:
        nltk.download('punkt_tab', quiet=True) 
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('rslp', quiet=True) 
        
        print("Recursos do NLTK prontos com sucesso!")
        
        db = next(get_db())
        all_companies_list = db.query(models.Empresa).all()
        
        if all_companies_list:
            search_engine_instance = SearchEngine(all_companies_list)
            print("Índice TF-IDF criado com sucesso!")
        
    except Exception as e:
        print(f"Erro na inicialização do NLTK/TF-IDF: {e}")
        raise RuntimeError(f"Falha na inicialização: {e}")

    print("Iniciando a criação das tabelas do banco de dados...")
    try:
        models.Base.metadata.create_all(bind=engine)
        print("Banco de dados e tabelas criadas com sucesso!")
    except Exception as e:
        print(f"Erro na criação das tabelas do BD: {e}")
        
    yield
    print("Aplicação encerrada.")


app = FastAPI(title="API de Pesquisa de Startups", lifespan=lifespan)


@app.post("/register", response_model=schemas.Token)
def register_user(user_data: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.Usuario).filter(models.Usuario.email == user_data.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="E-mail já cadastrado.")
    
    hashed_password = security.hash_password(user_data.password)
    new_user = models.Usuario(email=user_data.email, senha_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    access_token = security.create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token/json", response_model=schemas.Token)
def login_with_json(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.Usuario).filter(models.Usuario.email == user_data.email).first()
    if not user or not security.verify_password(user_data.password, user.senha_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/token", response_model=schemas.Token)
def login_with_form(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.Usuario).filter(models.Usuario.email == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.senha_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-mail ou senha incorretos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/companies", response_model=List[schemas.Empresa], status_code=status.HTTP_200_OK)
def list_all_companies(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(security.get_current_user)
):
    results = db.query(models.Empresa).all()
    if not results:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nenhuma empresa encontrada no sistema.")
    return results


@app.get("/optimized_search", response_model=List[schemas.Empresa], status_code=status.HTTP_200_OK)
def optimized_search_companies(
    query: str,
    fase: Optional[str] = None,
    current_user: schemas.User = Depends(security.get_current_user)
):
    global search_engine_instance
    
    if search_engine_instance is None:
        raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
             detail="O serviço de busca ainda não foi inicializado ou falhou ao carregar o índice."
        )

    results = search_engine_instance.optimized_search(query=query, fase=fase)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma startup encontrada com a sua pesquisa."
        )
    
    return results

if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)