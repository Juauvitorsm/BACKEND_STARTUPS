import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any
from fuzzywuzzy import fuzz
import inspect

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)
except Exception:
    pass

def initialize_nlp_resources():
    global stemmer, stop_words_pt
    try:
        stemmer = RSLPStemmer()
        stop_words_pt = set(stopwords.words('portuguese'))
    except LookupError:
        nltk.download('rslp', quiet=True)
        stemmer = RSLPStemmer()
        stop_words_pt = set(stopwords.words('portuguese'))
    
    CORP_AND_COMMON_STOP_WORDS = {'empresa', 'ltda', 's.a', 'eireli', 'companhia', 'solucoes', 'inovacao', 'tecnologia', 'group', 'grupo', 'de', 'a', 'o', 'e', 'do', 'da', 'dos', 'as', 'os', 
        'um', 'uma', 'uns', 'umas', 'para', 'na', 'no', 'em', 'por', 'foco', 'quer', 'busco', 'ramo', 'com', 'eu', 'tu', 'ele', 'ela', 'documento', 'fazer', 'quero', 'um', 'peca'}
    stop_words_pt.update(CORP_AND_COMMON_STOP_WORDS)
    
initialize_nlp_resources()

def custom_tokenizer(text):
    text = unidecode(text).lower()
    tokens = word_tokenize(text, language='portuguese')
    final_tokens = []
    for t in tokens:
        if t in stop_words_pt or len(t) <= 1:
            continue
        if t.isalpha():
            final_tokens.append(stemmer.stem(t))
        elif t.isalnum(): 
            final_tokens.append(t)
    return final_tokens


class SearchEngine:
    
    def __init__(self, all_companies_list: List[Any]):
        self.all_companies_list = all_companies_list
        self.tfidf_vectorizer = None
        self.company_vectors = None
        
        if self.all_companies_list:
            company_texts = [
                unidecode(f"{c.nome_da_empresa} {c.solucao} {c.setor_principal} {c.setor_secundario}").lower()
                for c in self.all_companies_list
            ]
            
            self.tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2))
            self.company_vectors = self.tfidf_vectorizer.fit_transform(company_texts)
        else:
             print("Aviso: SearchEngine inicializado sem dados.")

    def optimized_search(self, query: str, fase: str = None, limit: int = 5):
        if self.tfidf_vectorizer is None or self.company_vectors is None:
            return []

        normalized_query = unidecode(query).lower()
        query_vector = self.tfidf_vectorizer.transform([normalized_query])
        cosine_scores = cosine_similarity(query_vector, self.company_vectors).flatten()
        
        scored_companies = []
        RELEVANCE_THRESHOLD = 0.015

        for i, score in enumerate(cosine_scores):
            company = self.all_companies_list[i]
            
            if score < RELEVANCE_THRESHOLD: 
                continue
                
            if fase and company.fase_da_startup != fase:
                continue

            company_raw_text = f"{company.nome_da_empresa} {company.solucao} {company.setor_principal} {company.setor_secundario}"
            fuzzy_tolerance_score = fuzz.token_set_ratio(unidecode(company_raw_text).lower(), normalized_query)
            
            tf_idf_weighted = score * 400
            fuzzy_bonus = fuzzy_tolerance_score * 0.50 
            final_score = tf_idf_weighted + fuzzy_bonus
            
            if final_score > 45.0: 
                scored_companies.append({'company': company, 'score': final_score})
        
        scored_companies.sort(key=lambda x: x['score'], reverse=True)
        
        return [item['company'] for item in scored_companies[:limit]]