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

pricing_researcher = Agent(
    role="Pesquisador de Preços", 
    goal="Analise e determine o melhor preço para o produto {produto}. \
    Considere fatores de mercado, concorrência e percepção de valor.",
    backstory="Você é um especialista em pesquisa de preços e mercado. Você possui vasta \
    experiência em ajudar empresas a definir estratégias de precificação com base em dados.",
    allow_delegation=False, 
    verbose=True, 
    llm="groq/llama3-8b-8192" 
)

pricing_strategist = Agent(
    role="Estratégia de Preços", 
    goal="Desenvolver uma estratégia de precificação para o produto {produto}. \
    Inclua recomendações baseadas em pesquisa e análise de mercado.",
    backstory="Você tem experiência em criar e implementar estratégias de precificação para \
    produtos variados, ajudando empresas a maximizar lucros e satisfação do cliente.",
    allow_delegation=False, 
    verbose=True, 
    llm="groq/llama3-8b-8192" 
)

tool = SerperDevTool()

task_pricing_researcher = Task(
    description="Analise o mercado e colete dados sobre a faixa de preços para o produto {produto}. \
    Considere a concorrência e fatores que afetam a percepção de preço. \
    Forneça insights sobre o melhor preço para maximizar vendas e lucro.",
    expected_output=(
        "Um relatório detalhado com a análise de preços do produto {produto}. \
        Inclua referências a dados externos e sugestões de preços."
    ), 
    tools=[tool], 
    agent=pricing_researcher,
)

task_pricing_strategist = Task(
    description="Desenvolva uma estratégia de precificação para o produto {produto}. \
    Inclua recomendações sobre como posicionar o produto no mercado e ajustar o preço \
    conforme a demanda e feedback do consumidor.",
    expected_output=(
        "Um plano de estratégia de preços para o produto {produto} que inclua recomendações \
        práticas e possíveis ajustes de preços baseados em mercado."
    ), 
    tools=[tool], 
    agent=pricing_strategist,
)

crew = Crew(
    agents=[pricing_researcher, pricing_strategist],
    tasks=[task_pricing_researcher, task_pricing_strategist],
    verbose=True,
    max_rpm=25
)
