import pytest
from typing import Set, Dict, Any, List
from unidecode import unidecode

from backend.app.search_engine import SearchEngine
from backend.app import models as models_app


MOCK_COMPANIES_DATA = [
    {
        "id": 1, "nome_da_empresa": "AgroSense BR", 
        "solucao": "Plataforma de IA para otimização de logística agrícola", 
        "setor_principal": "Agrotech", "setor_secundario": "Inteligência Artificial", 
        "fase_da_startup": "Scale-up", "cnpj": "11.111.111/0001-11", 
        "endereco": "A", "ano_de_fundacao": 2018, "site": "a", "rede_social": "a", "cadastrado_por": "a", "cargo": "a", "email": "a", "colaboradores": "a", "publico_alvo": "a", "modelo_de_negocio": "a", "recebeu_investimento": "a", "negocios_no_exterior": "a", "faturamento": "a", "patente": "a", "ja_pivotou": "a", "comunidades": "a",
    },
    {
        "id": 2, "nome_da_empresa": "CyberGuard Pro", 
        "solucao": "Software de segurança proativa que utiliza machine learning", 
        "setor_principal": "Segurança da Informação", "setor_secundario": "SaaS", 
        "fase_da_startup": "Seed", "cnpj": "22.222.222/0001-22", 
        "endereco": "A", "ano_de_fundacao": 2020, "site": "a", "rede_social": "a", "cadastrado_por": "a", "cargo": "a", "email": "a", "colaboradores": "a", "publico_alvo": "a", "modelo_de_negocio": "a", "recebeu_investimento": "a", "negocios_no_exterior": "a", "faturamento": "a", "patente": "a", "ja_pivotou": "a", "comunidades": "a",
    },
    {
        "id": 3, "nome_da_empresa": "ProtoMesh 3D", 
        "solucao": "Serviço de impressão 3D industrial e prototipagem rápida", 
        "setor_principal": "Manufatura Aditiva", "setor_secundario": "Engenharia", 
        "fase_da_startup": "Growth", "cnpj": "33.333.333/0001-33", 
        "endereco": "A", "ano_de_fundacao": 2017, "site": "a", "rede_social": "a", "cadastrado_por": "a", "cargo": "a", "email": "a", "colaboradores": "a", "publico_alvo": "a", "modelo_de_negocio": "a", "recebeu_investimento": "a", "negocios_no_exterior": "a", "faturamento": "a", "patente": "a", "ja_pivotou": "a", "comunidades": "a",
    },
]

MOCK_COMPANIES: List[models_app.Empresa] = [models_app.Empresa(**data) for data in MOCK_COMPANIES_DATA]


SEARCH_TEST_CASES = [
    {
        "query": "Plataforma de IA para fazendas",
        "description": "Teste 1: Busca AGROTECH (Deve achar #1)",
        "relevance_rules": {"setor_principal": ["Agrotech"], "solucao": ["logística agrícola"]},
    },
    {
        "query": "Segurança de dados e Machine Learning",
        "description": "Teste 2: Busca TECH (Deve achar #2)",
        "relevance_rules": {"setor_principal": ["Segurança da Informação"], "solucao": ["software", "machine learning"]},
    },
    {
        "query": "Impressão industrial com 3D",
        "description": "Teste 3: Busca 3D/Manufatura (Deve achar #3)",
        "relevance_rules": {"setor_principal": ["Manufatura Aditiva"], "solucao": ["impressão 3D", "prototipagem"]},
    },
    {
        "query": "Software Machine Learning",
        "description": "Teste 4: Busca por Software com Filtro de Fase Seed (Deve achar #2)",
        "fase_filter": "Seed",
        "relevance_rules": {"fase_da_startup": ["Seed"], "solucao": ["software", "machine learning"]},
    },
]

TEST_ENGINE = SearchEngine(MOCK_COMPANIES)


def calculate_metrics(retrieved_ids: list, expected_ids: Set[int]):
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    true_positives = retrieved_set.intersection(expected_set)
    tp_count = len(true_positives)
    retrieved_count = len(retrieved_set)
    expected_count = len(expected_ids)

    precision = tp_count / retrieved_count if retrieved_count else 0.0
    recall = tp_count / expected_count if expected_count else 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
        
    return precision, recall, f1_score, tp_count, retrieved_count, expected_count


def get_ground_truth_ids_direct(all_companies: List[models_app.Empresa], case: Dict[str, Any]) -> Set[int]:
    expected_ids = set()
    relevance_rules = case.get("relevance_rules", {})
    fase_filter = case.get("fase_filter")
    
    for company in all_companies:
        if fase_filter and company.fase_da_startup != fase_filter:
            continue
            
        is_relevant = False
        
        for field, keywords in relevance_rules.items():
            company_value = unidecode(getattr(company, field, '') or '').lower()
            
            if any(unidecode(keyword).lower() in company_value for keyword in keywords):
                is_relevant = True
                break
        
        if is_relevant:
            expected_ids.add(company.id)
            
    return expected_ids


@pytest.mark.parametrize("case", SEARCH_TEST_CASES, ids=[c["description"] for c in SEARCH_TEST_CASES])
def test_optimized_search_logic_metrics(case: Dict[str, Any]):
    
    expected_ids = get_ground_truth_ids_direct(TEST_ENGINE.all_companies_list, case)
    
    if not expected_ids:
        pytest.skip(f"Ground Truth VAZIO para o caso: {case['description']}. Pulando teste.")
        return 

    retrieved_companies = TEST_ENGINE.optimized_search(
        query=case['query'], 
        fase=case.get('fase_filter')
    )
    retrieved_ids = [company.id for company in retrieved_companies]
    
    precision, recall, f1_score, tp, retrieved_count, expected_count = calculate_metrics(retrieved_ids, expected_ids)

    print(f"\n--- Teste Lógico: {case['description']} ---")
    print(f"Query: {case['query']}")
    print(f"Total Relevantes (Ground Truth): {expected_count}")
    print(f"IDs Retornadas: {retrieved_ids}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    
    MIN_PRECISION = 0.60
    MIN_RECALL = 0.60
    
    assert precision >= MIN_PRECISION, (
        f"Precision baixa ({precision:.2f}). Esperado >= {MIN_PRECISION}."
    )
    assert recall >= MIN_RECALL, (
        f"Recall baixo ({recall:.2f}). Esperado >= {MIN_RECALL}."
    )