# Plano de finalização — Trabalho GB

## 1. Corrigir bug de output
- [ ] Investigar por que `model.predict` retorna 10 valores em vez de 5
- [ ] Garantir que o modelo carregado/definido tem `Dense(5, activation='sigmoid')`

---

## 2. Carregamento e divisão dos dados
- [ ] Carregar o dataset de treino (`sound_event/train`) para as 5 classes escolhidas
- [ ] Converter todos os áudios para mel-spectrogram (mesmo pipeline do pré-processamento atual)
- [ ] Dividir em conjuntos de treino e validação (ex: 80/20)
- [ ] Documentar em markdown: tamanho do dataset, distribuição de classes, duração média dos áudios

---

## 3. CNN treinada do zero
- [ ] Implementar arquitetura CNN from scratch com Keras
- [ ] Documentar em markdown: justificativa das escolhas (camadas, ativações, loss, otimizador)
- [ ] Treinar com os dados carregados (`model.fit`)

### Experimentos (mínimo 12)
Variar e registrar os resultados de:
- [ ] 3 variações de número de camadas (ex: 2, 4, 6 camadas convolucionais)
- [ ] 3 variações de neurônios por camada (ex: 32, 64, 128 filtros)
- [ ] 2 funções de ativação (ex: ReLU vs ELU)
- [ ] 2 funções de erro (ex: binary_crossentropy vs categorical_crossentropy)
- [ ] 2 otimizadores (ex: Adam vs SGD)

---

## 4. CNN com transfer learning (EfficientNetB0 — já iniciada)
- [ ] Corrigir o output para 5 classes
- [ ] Adicionar dados de treino e chamar `model.fit`
- [ ] Documentar em markdown: justificativa do modelo pré-treinado escolhido

### Experimentos (mínimo 12)
Variar e registrar:
- [ ] 3 variações de camadas densas no topo (ex: 1, 2, 3 Dense layers)
- [ ] 3 variações de neurônios por camada densa (ex: 128, 256, 512)
- [ ] 2 funções de ativação nas camadas densas (ex: ReLU vs tanh)
- [ ] 2 funções de erro (ex: binary_crossentropy vs focal loss)
- [ ] 2 otimizadores (ex: Adam vs RMSprop)

---

## 5. Tabelas de resultados
- [ ] Para cada rede, criar uma tabela markdown com colunas:
  `Experimento | Camadas | Neurônios | Ativação | Loss | Otimizador | AUC (val) | Loss (val)`
- [ ] Destacar a melhor configuração de cada rede
- [ ] Escrever parágrafo discutindo o impacto de cada hiperparâmetro

---

## 6. Relatório no estilo literate programming
Adicionar células markdown ao longo do notebook cobrindo:
- [ ] Introdução: descrição do problema e do dataset
- [ ] Pré-processamento: explicar mel-spectrogram, parâmetros usados e justificativa
- [ ] Arquitetura de cada rede: descrever e justificar cada escolha
- [ ] Resultados e conclusões: qual rede foi melhor e por quê

---

## 7. Vídeo de apresentação
- [ ] Gravar vídeo de até 3 minutos cobrindo: contextualização, arquiteturas, resultados
- [ ] Incluir link do vídeo no início do notebook (célula markdown de resumo)
