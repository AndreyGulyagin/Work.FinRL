#  https://habr.com/ru/companies/otus/articles/805867/
#  https://habr.com/ru/articles/505674/



import pandas as pd
import yfinance as yf


from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2019-01-01'
TEST_START_DATE = '2019-01-01'
TEST_END_DATE = '2024-09-07'


df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=config_tickers.FX_TICKER).fetch_data()

print(df.head(5))


# добавляем индикаторы


fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)
processed = fe.preprocess_data(df)

print(processed.head(5))
