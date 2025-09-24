from sqlalchemy import Column, Integer, String, Text, BigInteger
from .database import Base



class Empresa(Base):
    __tablename__ = "startups"
    __table_args__ = {'schema': 'public'}

    id = Column("id", BigInteger, primary_key=True, index=True)
    nome_da_empresa = Column("nome_da_empresa", String(255), index=True)
    endereco = Column("endereço", String)
    cnpj = Column("cnpj", String(18), unique=True, index=True)
    ano_de_fundacao = Column("ano_de_fundação", BigInteger)
    site = Column("site", String)
    rede_social = Column("rede_social", String)
    cadastrado_por = Column("cadastrado_por", String)
    cargo = Column("cargo", String)
    email = Column("e-mail", String)
    setor_principal = Column("setor_principal", String, index=True)
    setor_secundario = Column("setor_secundario", String)
    fase_da_startup = Column("fase_da_startup", String)
    colaboradores = Column("colaboradores", String)
    publico_alvo = Column("publico_alvo", String)
    modelo_de_negocio = Column("modelo_de_negocio", String)
    recebeu_investimento = Column("recebeu_investimento", String)
    negocios_no_exterior = Column("negócios_no_exterior", String)
    faturamento = Column("faturamento", String)
    patente = Column("patente", String)
    ja_pivotou = Column("já_pivotou?", String)
    comunidades = Column("comunidades", String)
    solucao = Column("solução", Text)



    def __repr__(self):
        return f"<Empresa(nome='{self.nome_da_empresa}', setor='{self.setor_principal}')>"

class Usuario(Base):
    __tablename__ = "usuarios"
    __table_args__ = {'schema': 'public'}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    senha_hash = Column(String(255))