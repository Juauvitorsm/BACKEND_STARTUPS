from pydantic import BaseModel, Field, EmailStr, field_validator
import re
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=4)

    @field_validator('email')
    def validate_email_domain(cls, v):
        if not v.endswith('@mti.com'):
            raise ValueError('O e-mail deve pertencer ao domínio @mti.com')
        return v
    
    @field_validator('password')
    def validate_password_complexity(cls, v):
        if not re.search(r'\d', v):
            raise ValueError('A senha deve conter pelo menos um número')
        return v
    
    
class UserLogin(BaseModel):
    email: str
    password: str
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    email: str

    class Config:
        from_attributes = True



class Empresa(BaseModel):
    id: int
    nome_da_empresa: str
    endereco: str
    cnpj: str
    ano_de_fundacao: int
    site: Optional[str] = None 
    rede_social: Optional[str] = None 
    cadastrado_por: Optional[str] = None 
    cargo: Optional[str] = None 
    email: Optional[str] = None 
    setor_principal: str
    setor_secundario: str
    fase_da_startup: str
    colaboradores: str 
    publico_alvo: str
    modelo_de_negocio: str
    recebeu_investimento: str 
    negocios_no_exterior: str 
    faturamento: str
    patente: str
    ja_pivotou: str 
    comunidades: str
    solucao: str

    class Config:
        from_attributes = True