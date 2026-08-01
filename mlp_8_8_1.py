import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

BASE = Path(__file__).parent
df = pd.read_csv(BASE / 'cumulative.csv', sep=',')

colunas_usadas = ['koi_duration', 'koi_depth', 'koi_steff', 'koi_model_snr', 'koi_score', 'koi_prad', 'koi_period']

# Tirando todas linhas que tem qualquer coluna usada == Null
df.dropna(subset=colunas_usadas, inplace=True)
df.reset_index(drop=True, inplace=True)

def z(x):
  media = np.mean(x)
  dp = np.std(x)
  return (x - media) / dp

# Colocando todas colunas usadas em escala reduzida
z_duration = z(df['koi_duration'])
z_depth = z(df['koi_depth'])
z_steff = z(df['koi_steff'])
z_model_snr = z(df['koi_model_snr'])
z_prad = z(df['koi_prad'])
z_period = df['koi_period']
y = df["koi_score"]
x = [z_duration, z_depth, z_steff, z_model_snr, z_prad, z_period]

# Separando set de treino e set de teste em 80% - 20%
x_np = np.array(x).T

x_train, x_test, y_train, y_test = train_test_split(x_np, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(x_train)}")
print(f"Test set size: {len(x_test)}")


x_train = list(x_train)
x_test = list(x_test)

y_train = list(y_train)
y_test = list(y_test)

class Neuronio:
    def __init__ (self, N_entradas):
        self.n_entradas = N_entradas
        self.eta = 0.01
        self.pesos = [rd.uniform(-0.5, 0.5) for _ in range(N_entradas + 1)]
        self.y = 0
        self.delta = 0

    def y_chapeu(self, entradas):
        # Calculando bias * peso_bias
        calc_bias = 1 * self.pesos[0]
        soma = []

        # Calculando somatoria de entrada * peso_entrada
        for i in range(self.n_entradas):
            soma.append(entradas[i] * self.pesos[i+1])

        # Somando os dois
        u = sum(soma) + calc_bias

        # Limitando valor de u para evitar overflow
        u = np.clip(u, -500, 500)

        # Calculando y_chapeu via sigmoide
        self.y = 1 / (1 + np.exp(-u))
        return self.y

    def calc_erro(self, saida_real):
        E = 0.5 * (saida_real - self.y)**2
        return E

    def delta_Lsaida(self, saida_real):
        self.delta = (self.y - saida_real) * self.y * (1 - self.y)
        return self.delta

    def delta_Loculta(self, listas_deltas_superiores, listas_pesos_superiores):
        soma_erros = sum(d * p for d, p in zip(listas_deltas_superiores, listas_pesos_superiores))
        self.delta = self.y * (1 - self.y) * soma_erros
        return self.delta

    def at_peso(self, entrada):
        # Atualizando peso do bias
        self.pesos[0] = self.pesos[0] - self.eta * self.delta * 1

        # Atualizando o resto dos pesos
        for i in range(self.n_entradas):
            self.pesos[i+1] = self.pesos[i+1] - self.eta * self.delta * entrada[i]

def set_layer(n_neuronio, n_entradas):
  layer = [Neuronio(n_entradas) for i in range(n_neuronio)]
  return layer

# Definindo layer`s

layer_1 = set_layer(8, 6)

layer_2 = set_layer(8, 8)

layer_output = set_layer(1, 8)

# Treinamento
epocas = 100
epoca = 0
erros_por_ciclo = []
erros_por_epoca = []

for j in range(epocas):
  l1_y_chap = []
  l2_delta = []
  l2_y_chap = []
  contas = 0

  while contas < len(y_train):

# --- Forward Pass ---

      for i in layer_1:
          l1_y_chap.append(i.y_chapeu(x_train[contas]))

      for i in layer_2:
          l2_y_chap.append(i.y_chapeu(l1_y_chap))

      layer_output[0].y_chapeu(l2_y_chap)

# --- Cálculo dos Erros (Deltas - Backward Pass) ---

      # 1. Delta da Saída
      layer_output[0].calc_erro(y_train[contas])
      del_saida = layer_output[0].delta_Lsaida(y_train[contas])

      # 2. Deltas da Camada 2
      for i in range(len(layer_2)):
          l2_delta.append(layer_2[i].delta_Loculta([del_saida], [layer_output[0].pesos[i + 1]]))

      # 3. Deltas da Camada 1
      l1_delta = []
      for i in range(len(layer_1)):
          pesos_conectados_a_mim = [neuronio_l2.pesos[i + 1] for neuronio_l2 in layer_2]


          l1_delta.append(layer_1[i].delta_Loculta(l2_delta, pesos_conectados_a_mim))


# --- Atualização dos Pesos ---

      layer_output[0].at_peso(l2_y_chap)

      for i in range(len(layer_2)):
          layer_2[i].at_peso(l1_y_chap)

      for i in range(len(layer_1)):
          layer_1[i].at_peso(x_train[contas])

# --- Prepara para a próxima iteração e armazenando erro ---

      erros_por_ciclo.append(layer_output[0].calc_erro(y_train[contas]))
      contas += 1

      l1_y_chap.clear()
      l2_y_chap.clear()
      l2_delta.clear()

# --- Armazena o erro medio da época

  erros_por_epoca.append(sum(erros_por_ciclo)/len(erros_por_ciclo))
  epoca += 1
  erros_por_ciclo.clear()

print(f"Training finish \n Total epochs {epoca}")

# Plotando gráfico

plt.figure(figsize=(14, 6))
plt.plot(range(1, epoca + 1), erros_por_epoca, marker='o', linestyle='-', color='blue')
plt.title('Erro Médio por Época de Treinamento')
plt.xlabel('Época')
plt.ylabel('Erro Médio')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# TESTE DA REDE NEURAL

acertos = 0
erros = 0
erro_total_teste = 0
lista_prev = []
lista_real = []

print("Initiating tests...\n")

for contas in range(len(y_test)):

    l1_y_chap = []
    for i in layer_1:
        l1_y_chap.append(i.y_chapeu(x_test[contas]))

    l2_y_chap = []
    for i in layer_2:
        l2_y_chap.append(i.y_chapeu(l1_y_chap))

    previsao_prob = layer_output[0].y_chapeu(l2_y_chap)

    erro_total_teste += layer_output[0].calc_erro(y_test[contas])


    previsao = 1 if previsao_prob >= 0.5 else 0

    valor_real = 1 if y_test[contas] >= 0.5 else 0


    if previsao == valor_real:
        acertos += 1
    else:
        erros += 1

    lista_prev.append(previsao)
    lista_real.append(valor_real)

# RESULTADOS FINAIS
total_amostras = len(y_test)
taxa_acerto = (acertos / total_amostras) * 100
erro_medio_teste = erro_total_teste / total_amostras

print("-" * 35)
print("TEST RESULTS")
print("-" * 35)
print(f"Total analized planets: {total_amostras}")
print(f"Correct answers: {acertos}")
print(f"Errors: {erros}")
print(f"Acurácia: {taxa_acerto:.2f}%")
print(f"Loss: {erro_medio_teste:.4f}")

# Matriz de confusão

print("")
print("-" * 35)
print()

cm = confusion_matrix(lista_real, lista_prev)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Falso', 'Verdadeiro'])
plt.xlabel('Forecasted Value')
plt.ylabel('Real Value')
plt.title('Confusion Matrix')
plt.show()

# Relatório de classificação

print("")
print("-" * 35)
print()

print('Classification Report:')
print(classification_report(lista_real, lista_prev))

# Calculo de especificidade

print("")
print("-" * 35)
print()

TN = cm[0, 0]
FP = cm[0, 1]

espec = TN / (TN + FP)
print(f'Specificity: {espec:.2f}')
