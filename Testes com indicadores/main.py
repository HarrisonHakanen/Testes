import Funcoes
import pandas as pd



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #info = GetAcoes(pd.read_csv("acoes_tickers.csv"), 0)
    info = GetAcoes(["PETR4.SA","ITSA4.SA","CVCB3.SA"],0)

    valor_previsto_lstm = prever_valor_ltsm(pd.DataFrame(info[0].baixas["Close"]), 20, 1, 180)

    valor_previsto_media = prever_valor_media(info[0].baixas, 10)
