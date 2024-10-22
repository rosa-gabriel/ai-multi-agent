import os

from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool
from litellm import completion

response = completion(
    model="groq/llama3-8b-8192",
    messages=[
        {"role": "user", "content": "hello"}
        ],
)
print(response)

from google.colab import drive
drive.mount('/content/drive')

market_researcher = Agent(
    role="Pesquisador de Mercado Sênior", 
    goal = "Garanta que o negócio {ideia} seja respaldado por pesquisas e \
    dados sólidos. Realize uma pesquisa abrangente e realista para a ideia de negócio \
    {ideia}. Forneça insights da sua pesquisa para o empreendedor.", 
    backstory="Você é um especialista de mercado habilitado para pesquisas de mercado e \
    muito habilidoso em validar ideias de negócios. Você trabalhou com várias empresas \
    estabelecidas.", 
    allow_delegation=False, 
    verbose=True, 
    llm="groq/llama3-8b-8192" 
)

enterpreneur_agent = Agent(
    role="Empreendedor experiente", 
    goal = "Criar um plano de marketing e um plano de negócios para {ideia}", 
    backstory="Você construiu empresas de sucesso. Você tem habilidade de \
    criar novas ideias de negócios e planos de marketing.", 
    allow_delegation=False, 
    verbose=True, 
    llm="groq/llama3-8b-8192" 
)

tool = SerperDevTool()

task_market_researcher = Task(
    description= " Analise os pontos fortes, fracos, oportunidades e ameaças da ideia de \
    negócio {ideia}"\
    "Estimar o tamanho do mercado  e o potencial de crescimento para essa ideia. \
    Avalie a viabilidade do modelo de negócios. \
    Avalie a existência de outras empresas com a mesma ideia no mercado. \
    Forneça insights para a criação do plano de negócios.", 
    expected_output=(
        "Um relatório detalhado de pesquisa de mercado para a ideia mencionada {ideia}. \
        Inclua as referências a dados externos para análise de mercado."
    ), 
    tools=[tool], 
    agent=market_researcher,
)

task_enterpreneur = Task(
    description= "Crie o plano de marketing e o plano de negócios para a {ideia} \
    Garanta que não haja discrepância entre os planos gerados. \
    Verifique se todos os conceitos importantes de planos de negócio e de \
    marketing foram cobertos. ", 
    expected_output=(
        "A saída deve conter um plano de negócios final para a {ideia}\
        E um plano de marketing final para a {ideia}."
    ), 
    tools=[tool], 
    agent=enterpreneur_agent,
    output_file='analise.md'
)

crew = Crew(
    agents= [market_researcher, enterpreneur_agent],
    tasks=[task_market_researcher, task_enterpreneur],
    verbose = True,
    max_rpm = 25
)

crew = Crew(
    agents= [market_researcher, enterpreneur_agent],
    tasks=[task_market_researcher, task_enterpreneur],
    verbose = True,
    max_rpm = 25
)
