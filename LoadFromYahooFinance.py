#  https://habr.com/ru/companies/otus/articles/805867/
#  https://habr.com/ru/articles/505674/



import pandas as pd
#import numpy as np
import yfinance as yf


from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl import config_tickers




#print(DOW_30_TICKER)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2019-01-01'
TEST_START_DATE = '2019-01-01'
TEST_END_DATE = '2024-09-07'

Path = '.\\'

myTickerList = pd.DataFrame({'NamePartOfFileName': ['DOW_30_TICKER',
                                                    'NAS_100_TICKER',
                                                    'SP_500_TICKER',
                                                    'HSI_50_TICKER',
#                                                    'SSE_50_TICKER',
#                                                    'CSI_300_TICKER',
                                                    'CAC_40_TICKER',
                                                    'DAX_30_TICKER',
                                                    'TECDAX_TICKER',
                                                    'MDAX_50_TICKER',
                                                    'SDAX_50_TICKER',
                                                    'LQ45_TICKER',
                                                    'SRI_KEHATI_TICKER',
                                                    'FX_TICKER'
                                                   ],
                             'NameTickerList': [config_tickers.DOW_30_TICKER,
                                                config_tickers.NAS_100_TICKER,
                                                config_tickers.SP_500_TICKER,
                                                config_tickers.HSI_50_TICKER,
#                                                config_tickers.SSE_50_TICKER,
#                                                config_tickers.CSI_300_TICKER,
                                                config_tickers.CAC_40_TICKER,
                                                config_tickers.DAX_30_TICKER,
                                                config_tickers.TECDAX_TICKER,
                                                config_tickers.MDAX_50_TICKER,
                                                config_tickers.SDAX_50_TICKER,
                                                config_tickers.LQ45_TICKER,
                                                config_tickers.SRI_KEHATI_TICKER,
                                                config_tickers.FX_TICKER
                                              ]
                            })


# загружаем данные


CounterListDescription = 0

while CounterListDescription < len(myTickerList):
    FilePathName = Path + myTickerList.loc[CounterListDescription, 'NamePartOfFileName'] + '.csv'
#    FilePathName = Path + myTickerList.loc[CounterListDescription, 'NamePartOfFileName'] + '.xlsx'
    print(FilePathName)
    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=myTickerList.loc[CounterListDescription, 'NameTickerList']).fetch_data()
#    df.to_excel(FilePathName)
    df.to_csv(FilePathName, index=False)
    CounterListDescription = CounterListDescription + 1


#df.to_excel ('mydata.xlsx')