import math
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import datetime
import dateutil.parser
import math


class Acao:

    def _init_(self, ticker: str, altas: list, baixas: list, info: list, P_B):
        self.ticker = ticker
        self.altas = altas
        self.baixas = baixas
        self.info = info


def create_df(df, steps=1):
    dataX, dataY = [], []

    for i in range(len(df) - steps - 1):
        a = df[i:(i + steps), 0]
        dataX.append(a)
        dataY.append(df[i + steps, 0])
    return np.array(dataX), np.array(dataY)


def prever_valor_media(array, dias_anteriores):
    diferenca_pelo_min = array["Close"].tail(dias_anteriores).mean() - array["Close"].tail(dias_anteriores).min()
    diferenca_pelo_max = array["Close"].tail(dias_anteriores).mean() - array["Close"].tail(dias_anteriores).max()
    media = array["Close"].tail(dias_anteriores).mean()

    ate = 0

    if (diferenca_pelo_min < 0):
        diferenca_pelo_min *= -1

    if (diferenca_pelo_max < 0):
        diferenca_pelo_max *= -1

    if (diferenca_pelo_max < diferenca_pelo_min):

        ate = media - diferenca_pelo_max

    else:

        ate = media - diferenca_pelo_min

    return [media, ate]


# PREVE O VALOR DA PRÓXIMA ALTA OU DA PRÓXIMA BAIXA, DEPENDE DO ARRAY QUE SERÁ PASSADO
def prever_valor_ltsm(array, dias_anteriores, dias_previsao, epocas):
    dias_retorno = dias_anteriores

    dias_previsao = dias_previsao

    qtd_linhas = len(array)

    qtd_linhas_treino = round(.70 * qtd_linhas)

    qtd_linhas_teste = qtd_linhas - qtd_linhas_treino

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(array)

    # Separa em treino e teste

    train = df_scaled[:qtd_linhas_treino]
    test = df_scaled[qtd_linhas_treino:qtd_linhas_treino + qtd_linhas_teste]

    # gerando dados de treino e de teste

    steps = dias_retorno

    X_train, Y_train = create_df(train, steps)
    X_teste, Y_teste = create_df(test, steps)

    # print('Treino x: ',X_train.shape,' Treino y: ',Y_train.shape)
    # print('Teste x: ',X_teste.shape,' Teste y: ',Y_teste.shape)

    """Esse 1 significa a quantidade de features o modelo tem"""
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_teste = X_teste.reshape(X_teste.shape[0], X_teste.shape[1], 1)

    # montando a rede
    model = Sequential()
    model.add(LSTM(35, return_sequences=True, input_shape=(steps, 1)))
    model.add(LSTM(35, return_sequences=True))
    model.add(LSTM(35))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    # model.summary()

    validation = model.fit(X_train, Y_train, validation_data=(X_teste, Y_teste), epochs=epocas, batch_size=dias_retorno,
                           verbose=2)

    length_test = len(test)

    days_input_steps = length_test - steps
    # print("Tamanho do test: ",length_test)
    # print("Quantidade de dias retornados: ",days_input_steps)

    input_steps = test[days_input_steps:]
    input_steps = np.array(input_steps).reshape(1, -1)
    # input_steps

    # Transformar em lista
    list_output_steps = list(input_steps)
    list_output_steps = list_output_steps[0].tolist()
    # list_output_steps

    pred_output = []
    i = 0
    n_future = 1

    while (i < n_future):
        if (len(list_output_steps) > steps):
            input_steps = np.array(list_output_steps[1:])
            print("{} dia. Valores de entrada -> {}".format(i, input_steps))
            input_steps = input_steps.reshape(1, -1)
            input_steps = input_steps.reshape((1, steps, 1))
            pred = model.predict(input_steps, verbose=0)
            print("{} dia. Valor previsto {}".format(i, pred))
            list_output_steps.extend(pred[0].tolist())
            list_output_steps = list_output_steps[1:]
            pred_output.extend(pred.tolist())
            i = i + 1
        else:
            input_steps = input_steps.reshape((1, steps, 1))
            pred = model.predict(input_steps, verbose=0)
            print(pred[0])
            list_output_steps.extend(pred[0].tolist())
            print(len(list_output_steps))
            pred_output.extend(pred.tolist())
            i = i + 1

    # print(pred_output)

    prev = scaler.inverse_transform(pred_output)
    prev = np.array(prev).reshape(1, -1)
    proxima_alta = list(prev)
    proxima_alta = prev[0].tolist()

    return proxima_alta


def High_low2(acao_df, h_l):
    acao_df['Date'] = pd.to_datetime(acao_df['Date'], format='%Y-%m-%d')

    acao_df = acao_df.set_index(pd.DatetimeIndex(acao_df['Date']))

    i = 0

    anos = list()
    meses = list()
    while (i < len(acao_df)):
        anos.append(acao_df["Date"][i].year)
        meses.append(acao_df["Date"][i].month)
        i += 1

    anos = list(dict.fromkeys(anos))
    meses = list(dict.fromkeys(meses))

    anos.sort(reverse=False)
    meses.sort(reverse=False)

    i = 0
    i_ano = 0
    i_mes = 0

    controlador = 0

    dados = pd.DataFrame()

    while (i < len(acao_df)):

        while (i_ano < len(anos)):

            while (i_mes < len(meses)):

                if (controlador == 0):

                    if (h_l == 0):
                        close = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            "Close"].min()
                        date = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            acao_df["Close"] == close].index.values

                        # dados = [[close]]
                        dados = [[close, date]]

                    elif (h_l == 1):
                        close = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            "Close"].max()
                        date = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            acao_df["Close"] == close].index.values

                        # dados = [[close]]
                        dados = [[close, date]]

                    controlador = 1
                else:

                    if (h_l == 0):
                        close = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            "Close"].min()
                        date = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            acao_df["Close"] == close].index.values

                    elif (h_l == 1):
                        close = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            "Close"].max()
                        date = \
                        acao_df[str(anos[i_ano]) + "-" + str(meses[i_mes]):str(anos[i_ano]) + "-" + str(meses[i_mes])][
                            acao_df["Close"] == close].index.values

                    if (not math.isnan(close)):

                        if (len(date) > 1):
                            date2 = date[0]

                            date = list()
                            date = [date2]

                        dados.append([close, date])
                        # dados.append([close])

                i_mes += 1

            i_mes = 0
            i_ano += 1

        i += 1
    dados = pd.DataFrame(dados, columns=['Close', 'Date'])

    return dados


def dados_lucro(altas_np, baixas_np):
    tamanho = 0
    diferencas = list()
    datas = list()

    data_compra = list()
    data_venda = list()
    valor_compra = list()
    valor_venda = list()
    lucro = list()

    if (len(altas_np) == len(baixas_np)):
        tamanho = len(altas_np)

    i = 0

    controlador = 0

    primeiro_ciclo = 0

    while (i < tamanho):

        if (len(baixas_np[i][1]) == 1):

            if (primeiro_ciclo == 0):

                b = baixas_np[i][1][0]

                vlr_c = baixas_np[i][0]

                a = altas_np[i][1][0]

                primeiro_ciclo += 1

            else:

                if (controlador == 0):
                    b = baixas_np[i][1][0]
                    vlr_c = baixas_np[i][0]
                    a = altas_np[i][1][0]
                else:
                    a = altas_np[i][1][0]

            if (b < a):

                lucro.append(altas_np[i][0] - vlr_c)

                data_venda.append(altas_np[i][1][0])

                valor_venda.append(altas_np[i][0])

                data_compra.append(b)

                valor_compra.append(vlr_c)

                controlador = 0
            else:

                controlador = 1

        i += 1

    dados = list(zip(data_compra, valor_compra, data_venda, valor_venda, lucro))

    negocios = pd.DataFrame(dados, columns=["Data compra", "Valor compra", "Data venda", "Valor venda", "Lucro"])

    return negocios


def GetAcoes(tickers, todas_acoes):
    contador = 0

    lista_de_acoes = list()

    cont = 0

    if (todas_acoes == 0):
        tickers = pd.DataFrame(tickers, columns=["Acoes"])

    for ticker in tickers["Acoes"]:

        if (contador < 10):

            print(ticker)

            acao = yf.download(ticker)
            acao_df = acao.rename_axis('Date').reset_index()

            acao_df = acao_df[['Date', 'Close']]

            altas_df = High_low2(acao_df, 1)
            baixas_df = High_low2(acao_df, 0)

            altas_np = altas_df.to_numpy()
            baixas_np = baixas_df.to_numpy()

            acao = Acao()
            acao.ticker = ticker
            acao.altas = altas_df
            acao.baixas = baixas_df
            acao.info = dados_lucro(altas_np, baixas_np)

            lista_de_acoes.append(acao)

            # contador += 1
        else:
            break

    return lista_de_acoes


def PesquisaAcao(acao_pesquisada, info):
    i = 0
    achou = 0
    tail = 0
    acao_pesquisada_info = pd.DataFrame()

    if (tail < len(info)):

        while (i < len(info)):

            if (info[i][1] == acao_pesquisada):
                acao_pesquisada_info = info[i][0]

                acao_pesquisada_info["Taxa retorno"] = np.log(
                    acao_pesquisada_info["Valor venda"] / acao_pesquisada_info["Valor compra"]) * 100

                achou = 1

            if (achou == 1):
                break
            i += 1
    else:

        print("O valor de tail deve ser menor que ", len(info))

    return acao_pesquisada_info