# water-classifier-hgbc

🎓 Contexto Acadêmico

Universidade: Universidade do Estado da Bahia (UNEB)

Curso: Engenharia de Software

Matéria: Aprendizado de Máquina em IA - Machine Learning

Docente: Marcos Figueredo

📄 Resumo do Projeto
Este projeto consiste na implementação e otimização de um classificador para determinar a potabilidade da água (Potability) a partir de dados físico-químicos contidos no dataset water.csv. O trabalho segue um protocolo rigoroso de validação cruzada e busca de hiperparâmetros, conforme exigido pelo roteiro da atividade.

Modelo Escolhido: HistGradientBoostingClassifier (HGBC).

Metodologia: Stratified K-Fold (k=5) e Randomized Search, com foco na otimização da métrica ROC-AUC.

📁 Estrutura do Repositório
O projeto segue a estrutura modular para garantir a separação de responsabilidades e a execução via CLI (Command Line Interface):

```
water-classifier-hgbc/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ data/
│  └─ raw/water.csv   <-- Dataset obrigatório
├─ src/
│  ├─ config.py       <-- Constantes e Configurações
│  ├─ preprocess.py   <-- Imputação por Mediana
│  ├─ model.py        <-- Definição do Pipeline (Imputer + HGBC)
│  ├─ train_cv.py     <-- Treinamento e Busca de Hiperparâmetros
│  └─ evaluate.py     <-- Avaliação Final e Geração de Figuras
├─ artifacts/          <-- Saída: Modelo treinado (best.pkl)
├─ figures/            <-- Saída: Gráficos (ROC, PR, CM, Importância)
└─ reports/            <-- Saída: Métricas e Resultados de CV
```

⚙️ Requisitos e Instalação
O projeto requer Python 3.8+ e as bibliotecas listadas no requirements.txt.

Clone o repositório:

```bash
git clone https://github.com/albierygs/water-classifier-HistGradientBoosting.git
cd water-classifier-HistGradientBoosting
```

Crie um ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

🚀 Protocolo de Execução (Reprodutibilidade)
A execução do projeto é realizada via linha de comando (CLI), com a semente fixa (random_state=42) garantindo que os resultados sejam 100% replicáveis.

1. Treinamento e Otimização (CV)
   Este comando executa a validação cruzada estratificada e a busca de hiperparâmetros (Randomized Search), salvando o modelo otimizado em artifacts/best.pkl.

   ```bash
   python -m src.train --data data/raw/water.csv --out reports/ --k 5 --seed 42
   ```

2. Avaliação Final e Geração de Figuras
   Este comando carrega o modelo otimizado e realiza a avaliação no conjunto de teste hold-out, salvando as métricas finais em reports/results.csv e todas as figuras exigidas em figures/.

   ```bash
   python -m src.evaluate --model artifacts/best.pkl --data data/raw/water.csv --out figures/
   ```

📊 Resultados do Melhor Modelo
O modelo HGBC foi otimizado utilizando o Stratified K-Fold. O melhor desempenho e a avaliação final demonstram a alta capacidade preditiva do classificador:

| Métrica  | Desempenho em CV (μ±σ) | Desempenho no Teste Final |
| -------- | ---------------------- | ------------------------- |
| ROC-AUC  | 0.9327±0.0082          | 0.9842                    |
| F1-Macro | (Não otimizado em CV)  | 0.9344                    |
| Accuracy | (Não otimizado em CV)  | 0.9344                    |

O baixo desvio padrão em CV (σ=0.0082) confirma que a variância do modelo foi efetivamente controlada pela regularização de hiperparâmetros, resultando em um classificador estável e confiável.

📜 Licença
Este projeto está licenciado sob os termos da licença MIT, conforme detalhado no arquivo LICENSE na raiz do repositório.
