import uvicorn
import nltk
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fuzzywuzzy import fuzz
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .database import engine, Base, get_db
from . import models, security, schemas
from .schemas import UserLogin


stemmer = None
stop_words_pt = None
tfidf_vectorizer = None
company_vectors = None
all_companies_list = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tfidf_vectorizer, company_vectors, all_companies_list
    
    print("Baixando e inicializando recursos do NLTK...")
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('rslp')
        
        global stemmer, stop_words_pt
        stemmer = nltk.stem.RSLPStemmer()
        stop_words_pt = set(stopwords.words('portuguese'))
        print("Recursos do NLTK prontos com sucesso!")
        
        db = next(get_db())
        all_companies_list = db.query(models.Empresa).all()
        
        if all_companies_list:
            company_texts = [
                unidecode(f"{c.nome_da_empresa} {c.solucao} {c.setor_principal} {c.setor_secundario}").lower()
                for c in all_companies_list
            ]
            
            def custom_tokenizer(text):
                tokens = word_tokenize(text, language='portuguese')
                return [t for t in tokens if t not in stop_words_pt and t.isalpha()] 

            tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
            company_vectors = tfidf_vectorizer.fit_transform(company_texts)
            print("Índice TF-IDF criado com sucesso!")
        
    except Exception as e:
        print(f"Erro na inicialização do NLTK/TF-IDF: {e}")
        raise RuntimeError("Falha na inicialização.")

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

@app.get("/search", response_model=List[schemas.Empresa], status_code=status.HTTP_200_OK)
def search_companies(
    query: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(security.get_current_user)
):
    all_companies = db.query(models.Empresa).all()
    if not all_companies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma empresa encontrada no sistema."
        )

    scored_companies = []
    normalized_query = unidecode(query).lower()
    query_tokens = [
        stemmer.stem(token)
        for token in word_tokenize(normalized_query, language='portuguese')
        if token and token not in stop_words_pt
    ]
    
    if not query_tokens:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sua pesquisa não contém palavras relevantes."
        )

    for company in all_companies:
        search_text = f"{company.nome_da_empresa} {company.solucao} {company.setor_principal} {company.setor_secundario}"
        
        normalized_search_text = unidecode(search_text).lower()
        search_tokens = [
            stemmer.stem(token)
            for token in word_tokenize(normalized_search_text, language='portuguese')
            if token and token not in stop_words_pt
        ]
        
        match_score = 0
        for q_token in query_tokens:
            for s_token in search_tokens:
                token_fuzz_score = fuzz.ratio(q_token, s_token)
                if token_fuzz_score > 80:
                    match_score += token_fuzz_score
        
        overall_score = fuzz.token_set_ratio(
            ' '.join(search_tokens), ' '.join(query_tokens)
        )
        final_score = (match_score + overall_score) / 2 if (match_score + overall_score) > 0 else 0
        
        if final_score > 70:
            scored_companies.append({'company': company, 'score': final_score})
    
    scored_companies.sort(key=lambda x: x['score'], reverse=True)
    
    results = [item['company'] for item in scored_companies]

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma startup encontrada com a sua pesquisa."
        )
    
    return results


@app.get("/filtered_search", response_model=List[schemas.Empresa], status_code=status.HTTP_200_OK)
def filtered_search_companies(
    query: str,
    fase: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(security.get_current_user)
):
    all_companies = db.query(models.Empresa).all()
    if not all_companies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma empresa encontrada no sistema."
        )

    scored_companies = []
    normalized_query = unidecode(query).lower()

    if not normalized_query.strip() and fase:

        filtered_companies_by_phase = [
            comp for comp in all_companies if comp.fase_da_startup == fase
        ]
        if not filtered_companies_by_phase:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Nenhuma startup encontrada com a sua pesquisa."
            )
        return filtered_companies_by_phase

    for company in all_companies:
        search_text = f"{company.nome_da_empresa} {company.solucao} {company.setor_principal} {company.setor_secundario}"
        normalized_search_text = unidecode(search_text).lower()
        
        final_score = fuzz.token_set_ratio(normalized_search_text, normalized_query)
        
        if final_score > 70:
            scored_companies.append({'company': company, 'score': final_score})
    
    scored_companies.sort(key=lambda x: x['score'], reverse=True)
    

    if fase:
        scored_companies = [item for item in scored_companies if item['company'].fase_da_startup == fase]

    results = [item['company'] for item in scored_companies]

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma startup encontrada com a sua pesquisa."
        )
    
    return results


@app.get("/optimized_search", response_model=List[schemas.Empresa], status_code=status.HTTP_200_OK)
def optimized_search_companies(
    query: str,
    fase: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(security.get_current_user)
):
    global tfidf_vectorizer, company_vectors, all_companies_list
    
    if tfidf_vectorizer is None or company_vectors is None or all_companies_list is None:
        return [c for c in db.query(models.Empresa).limit(5).all()] 

    normalized_query = unidecode(query).lower()
    
    query_vector = tfidf_vectorizer.transform([normalized_query])
    
    cosine_scores = cosine_similarity(query_vector, company_vectors).flatten()
    
    scored_companies = []
    
    for i, score in enumerate(cosine_scores):
        company = all_companies_list[i]
        
        if score < 0.1: 
            continue
            
        if fase and company.fase_da_startup != fase:
            continue

        company_raw_text = f"{company.nome_da_empresa} {company.solucao} {company.setor_principal} {company.setor_secundario}"
        
        fuzzy_tolerance_score = fuzz.token_set_ratio(unidecode(company_raw_text).lower(), normalized_query)
        
        tf_idf_weighted = score * 100 
        fuzzy_bonus = fuzzy_tolerance_score * 0.15
        
        final_score = tf_idf_weighted + fuzzy_bonus
        
        scored_companies.append({'company': company, 'score': final_score})
    
    scored_companies.sort(key=lambda x: x['score'], reverse=True)
    
    results = [item['company'] for item in scored_companies[:5]]

    return results


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)