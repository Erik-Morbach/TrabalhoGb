# Revisão dos passos 1, 2 e 3

Revisão do código adicionado ao `Relatório final.ipynb` nas etapas de correção de bug (1), carregamento/divisão dos dados (2) e CNN do zero (3). Achados organizados por severidade.

---

## 🔴 Crítico — Inputs sem normalização (afeta passos 2 e 3) ✅ CORRIGIDO

> **Status:** aplicado em `cell-8` (dB + padronização) e documentado em `cell-7`.

Erro mais importante; explica o sintoma das predições degeneradas (~0.5).

Na `cell-5`, o `.npy` salvo é o **mel-spectrograma de potência cru** (`mel_spectrogram`), não o dB:

```python
mel_spectrogram = librosa.feature.melspectrogram(...)
mel_s_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # calculado mas NÃO usado
np.save(target_file_npy, mel_spectrogram)   # salva a POTÊNCIA crua
```

Os valores são minúsculos e com range enorme (ex.: `2.8e-07`, `9.0e-08`). Em `cell-8` o `X` é montado direto desses valores, **sem nenhuma normalização**.

### Consequências
1. **CNN do zero (passo 3):** convergência muito ruim — a rede precisa "aprender" a escala antes de aprender qualquer padrão.
2. **Transfer learning (passo 1/4):** motivo real das predições terem dado todas ~0.5. O `efficientnet.preprocess_input` + a normalização interna do EfficientNet esperam valores na faixa ~[0, 255]. Com inputs ~`1e-7`, tudo vira praticamente constante → o modelo não recebe sinal → saída degenerada.

### Correção
Converter para dB e padronizar. No `cell-8`:

```python
mel_db = librosa.power_to_db(mel, ref=np.max)               # ~[-80, 0] dB
mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6) # média 0, desvio 1
```

Resolve tanto a CNN do zero quanto o output degenerado do transfer learning de uma vez.

---

## 🟡 Médio

> **Status:** ambos aplicados. #1 → `GlobalAveragePooling2D` em `cell-12`. #2 → métricas da mesma época em `cell-13`/`cell-14`. Bônus: `tf.random.set_seed(42)` adicionado (resolve o item de seed da seção 🟢).

### 1. `Flatten` + `Dense` gera contagem de parâmetros enorme nas configs rasas (`cell-12`) ✅ CORRIGIDO
Com 2 camadas de pooling, sobram `32×32×filtros` features. Para `filters=128`: `32×32×128 = 131.072` → `Dense(128)` = **~16,8 milhões de parâmetros numa única camada**. A rede de transfer learning já usa `GlobalAveragePooling2D` — usar o mesmo aqui deixaria as redes comparáveis, mais leves e menos propensas a overfit:

```python
x = layers.GlobalAveragePooling2D()(x)   # no lugar de Flatten()
```

### 2. Reporte de métrica mistura épocas diferentes (`cell-13`) ✅ CORRIGIDO
```python
val_auc  = max(hist.history['val_auc'])
val_loss = min(hist.history['val_loss'])
```
O melhor AUC e o menor loss podem vir de **épocas diferentes** — não representam o mesmo estado do modelo. Para ser metodologicamente honesto, usar a métrica do mesmo ponto (ex.: `EarlyStopping(restore_best_weights=True)` e reportar a época restaurada, ou simplesmente a última época).

---

## 🟢 Menor

- **Filtros constantes em profundidade** (`cell-12`): o convencional é crescer (32→64→128). Como está, todas as camadas usam o mesmo `filters`. Não é erro, só incomum.
- **Sem seed** (`tf.random.set_seed`): os 12 experimentos não são reprodutíveis entre execuções — atrapalha a discussão comparativa.
- **Eixo temporal comprimido**: resize de `128×626` → `128×128` comprime o tempo ~5×. Aceitável para o trabalho, mas vale uma linha no relatório.
- **Treino em CPU é lento**: o log mostra "GPU will not be used". 12 redes × 20 épocas no CPU vai demorar muito — rodar no Colab com GPU (como o enunciado pede).

---

## Prioridade de correção

A correção de **maior impacto e menor esforço** é a normalização (🔴): uma única mudança no `cell-8` que conserta o treino da CNN do zero **e** o output degenerado do transfer learning. Em seguida, trocar `Flatten` por `GlobalAveragePooling2D` (🟡 #1).
