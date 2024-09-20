# устанавливаем библиотеки numpy pandas matplotlib finrl 'gymnasium==0.29.1' stable_baselines3
#                          alpaca_trade_api exchange_calendars stockstats
#                          wrds yfinance pyfolio gym 'shimmy>=0.2.1' tensorboard


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime
from os.path import exists

import warnings

warnings.filterwarnings('ignore')

# %matplotlib inline

from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from finrl.main import check_and_make_directories
from pprint import pprint
from stable_baselines3.common.logger import configure
import sys

# sys.path.append("../FinRL")

import pickle

import itertools

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

from finrl.config_tickers import DOW_30_TICKER

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

print(DOW_30_TICKER)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2019-01-01'
TEST_START_DATE = '2019-01-01'
TEST_END_DATE = '2021-01-01'

# загружаем данные
df = YahooDownloader(start_date=TRAIN_START_DATE,
                     end_date=TEST_END_DATE,
                     ticker_list=DOW_30_TICKER).fetch_data()

# посмотрим размер скачанных данных
df.shape
df.head()

# преобразуем дату
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# Отсортируем данные по дате и коду акции
df.sort_values(['date', 'tic'], ignore_index=True).head()

# Читаем финансовые данные
# url = 'https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv'

# fund = pd.read_csv(url)
fund = pd.read_csv('dow_30_fundamental_wrds.csv')

# Проверим корректность загруженных данных
fund.head()

# Список элементов, используемых для расчета финансовых показателей

items = [
    'datadate',  # Дата
    'tic',  # Тикер
    'oiadpq',  # Квартальная операционная прибыль
    'revtq',  # Квартальные доходы
    'niq',  # Квартальная чистая прибыль
    'atq',  # Всего активов
    'teqq',  # Собственный капитал
    'epspiy',  # Прибыль на акцию (базовая), включая внештатные элементы
    'ceqq',  # Общий капитал
    'cshoq',  # Общее количество обыкновенных акций
    'dvpspq',  # Дивиденды на акцию
    'actq',  # Оборотные активы
    'lctq',  # Оборотные пассивы
    'cheq',  # Денежные средства и их эквиваленты
    'rectq',  # Дебиторская задолженность
    'cogsq',  # Себестоимость проданных товаров
    'invtq',  # Инвентарь
    'apq',  # Кредиторская задолженность
    'dlttq',  # Долгосрочные обязательства
    'dlcq',  # Задолженность по текущим обязательствам
    'ltq'  # Пассивы
]

# Пропущены элементы, которые не будут использоваться
fund_data = fund[items]

# Переименование названий столбцов для удобства
fund_data = fund_data.rename(columns={
    'datadate': 'date',  # Дата
    'oiadpq': 'op_inc_q',  # Квартальная операционная прибыль
    'revtq': 'rev_q',  # Квартальные доходы
    'niq': 'net_inc_q',  # Квартальная чистая прибыль
    'atq': 'tot_assets',  # Активы
    'teqq': 'sh_equity',  # Собственный капитал
    'epspiy': 'eps_incl_ex',  # Прибыль на акцию (базовая), включая внештатные элементы
    'ceqq': 'com_eq',  # Общий капитал
    'cshoq': 'sh_outstanding',  # Общее количество обыкновенных акций
    'dvpspq': 'div_per_sh',  # Дивиденды на акцию
    'actq': 'cur_assets',  # Оборотные активы
    'lctq': 'cur_liabilities',  # Оборотные пассивы
    'cheq': 'cash_eq',  # Денежные средства и их эквиваленты
    'rectq': 'receivables',  # Дебиторская задолженность
    'cogsq': 'cogs_q',  # Себестоимость проданных товаров
    'invtq': 'inventories',  # Инвентарь
    'apq': 'payables',  # Кредиторская задолженность
    'dlttq': 'long_debt',  # Долгосрочные обязательства
    'dlcq': 'short_debt',  # Задолженность по текущим обязательствам
    'ltq': 'tot_liabilities'  # Пассивы
})

# Check the data
fund_data.head()

# Расчет финансовых показателей
date = pd.to_datetime(fund_data['date'], format='%Y%m%d')

tic = fund_data['tic'].to_frame('tic')

# Показатели рентабельности
# Операционная маржа (OPM)
OPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='OPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        OPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        OPM.iloc[i] = np.nan
    else:
        OPM.iloc[i] = np.sum(fund_data['op_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Чистая операционная маржа (NPM)
NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='NPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        NPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        NPM.iloc[i] = np.nan
    else:
        NPM.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Рентабельность активов (ROA)
ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROA')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROA[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROA.iloc[i] = np.nan
    else:
        ROA.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['tot_assets'].iloc[i]

# Рентабельность собственного капитала (ROE)
ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROE')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROE[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROE.iloc[i] = np.nan
    else:
        ROE.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['sh_equity'].iloc[i]

    # Для расчета коэффициентов оценки в следующем подразделе предварительно рассчитайте показатели на одну акцию
# Прибыль на акцию (EPS)
EPS = fund_data['eps_incl_ex'].to_frame('EPS')

# Собственный капитал на акцию (Book Value Per Share)
BPS = (fund_data['com_eq'] / fund_data['sh_outstanding']).to_frame('BPS')  # Need to check units

# Дивиденды на акцию (Dividends Per Share)
DPS = fund_data['div_per_sh'].to_frame('DPS')

# Коэффициенты ликвидности
# Коэффициент текущей ликвидности (Current Ratio)
cur_ratio = (fund_data['cur_assets'] / fund_data['cur_liabilities']).to_frame('cur_ratio')

# Быстрый коэффициент ликвидности (Quick Ratio)
quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities']).to_frame('quick_ratio')

# Коэффициент денежной ликвидности (Cash Ratio)
cash_ratio = (fund_data['cash_eq'] / fund_data['cur_liabilities']).to_frame('cash_ratio')

# Коэффициенты эффективности
# Коэффициент оборачиваемости запасов (Inventory Turnover Ratio)
inv_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='inv_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        inv_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        inv_turnover.iloc[i] = np.nan
    else:
        inv_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['inventories'].iloc[i]

# Коэффициент оборачиваемости дебиторской задолженности (Receivables Turnover Ratio)
acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_rec_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_rec_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_rec_turnover.iloc[i] = np.nan
    else:
        acc_rec_turnover.iloc[i] = np.sum(fund_data['rev_q'].iloc[i - 3:i]) / fund_data['receivables'].iloc[i]

# Коэффициент оборачиваемости кредиторской задолженности (Payable Turnover Ratio)
acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_pay_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_pay_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_pay_turnover.iloc[i] = np.nan
    else:
        acc_pay_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['payables'].iloc[i]

## Коэффициенты кредитного плеча (Leverage Financial Ratios)
# Коэффициент капитализации долга (Debt Capitalization Ratio)
debt_ratio = (fund_data['tot_liabilities'] / fund_data['tot_assets']).to_frame('debt_ratio')

# Коэффициент финансового плеча (Debt-to-Equity Ratio)
debt_to_equity = (fund_data['tot_liabilities'] / fund_data['sh_equity']).to_frame('debt_to_equity')

# Соберем датафрейм со всеми рассчитанными показателями
ratios = pd.concat([date, tic, OPM, NPM, ROA, ROE, EPS, BPS, DPS,
                    cur_ratio, quick_ratio, cash_ratio, inv_turnover, acc_rec_turnover, acc_pay_turnover,
                    debt_ratio, debt_to_equity], axis=1)

# Check the ratio data
print(ratios.head())

ratios.tail()

# Replace NAs infinite values with zero
final_ratios = ratios.copy()
final_ratios = final_ratios.fillna(0)
final_ratios = final_ratios.replace(np.inf, 0)

final_ratios.head()

final_ratios.tail()

list_ticker = df["tic"].unique().tolist()
list_date = list(pd.date_range(df['date'].min(), df['date'].max()))
combination = list(itertools.product(list_date, list_ticker))

# Merge stock price data and ratios into one dataframe
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"], how="left")
processed_full = processed_full.merge(final_ratios, how='left', on=['date', 'tic'])
processed_full = processed_full.sort_values(['tic', 'date'])

# Backfill the ratio data to make them daily
processed_full = processed_full.bfill(axis='rows')

processed_full['PE'] = processed_full['close'] / processed_full['EPS']
processed_full['PB'] = processed_full['close'] / processed_full['BPS']
processed_full['Div_yield'] = processed_full['DPS'] / processed_full['close']

# Удалим показатели на одну акцию, использованные для расчета коэффициентов.
processed_full = processed_full.drop(columns=['day', 'EPS', 'BPS', 'DPS'])

# Заменить пропуски нулями
processed_full = processed_full.copy()
processed_full = processed_full.fillna(0)
processed_full = processed_full.replace(np.inf, 0)

# посмотрим на финальный датасет для моделирования
print(processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10))

## 5.1 Разделим данные на train и test (trade)

train_data = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade_data = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)
# Check the length of the two datasets
print(len(train_data))
print(len(trade_data))

train_data.head()

trade_data.head()

## 5.2 Построим модель рынка

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


# matplotlib.use("Agg")

# from stable_baselines3.common import logger


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df,
            stock_dim,
            hmax,
            initial_amount,
            buy_cost_pct,
            sell_cost_pct,
            reward_scaling,
            state_space,
            action_space,
            tech_indicator_list,
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
            make_plots=False,
            print_verbosity=10,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                                self.state[index + 1]
                                * sell_num_shares
                                * (1 - self.sell_cost_pct)
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                                self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // self.state[index + 1]
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                        self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_trade_{}.png".format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                    self.state[0]
                    + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
                    - self.initial_amount
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:

            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.initial_amount]
                        + self.data.close.values.tolist()
                        + [0] * self.stock_dim
                        + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
                )
            else:
                # for single stock
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim
                        + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
                )
            else:
                # for single stock
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                    [self.state[0]]
                    + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum(
                [
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ],
                [],
            )
            )

        else:
            # for single stock
            state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            )
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


ratio_list = ['OPM', 'NPM', 'ROA', 'ROE', 'cur_ratio', 'quick_ratio', 'cash_ratio', 'inv_turnover', 'acc_rec_turnover',
              'acc_pay_turnover', 'debt_ratio', 'debt_to_equity',
              'PE', 'PB', 'Div_yield']

stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# Параметры среды (модели рынка)
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ratio_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

# Establish the training environment using StockTradingEnv() class
e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)

# создаем экземпляр окружения для тренировки модели

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# Часть 6: Тренировка агента

agent = DRLAgent(env=env_train)

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

### Обучим агента: 5 алгоритмов (A2C, DDPG, PPO, TD3, SAC)

### Model 1: PPO

agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

if if_using_ppo:
    # set up logger
    tmp_path = RESULTS_DIR + '/ppo'
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ppo.set_logger(new_logger_ppo)

    if exists('./trained_models/trained_ppo.model'):
        trained_ppo = model_ppo.load('./trained_models/trained_ppo.model')
else:
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=50000) if if_using_ppo else None

    trained_ppo.save('./trained_models/trained_ppo.model')

### Model 2: DDPG

agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")

if if_using_ddpg:
    # set up logger
    tmp_path = RESULTS_DIR + '/ddpg'
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_ddpg.set_logger(new_logger_ddpg)

if exists('./trained_models/trained_ddpg.model'):
    trained_ddpg = model_ddpg.load('./trained_models/trained_ddpg.model')
else:
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000) if if_using_ddpg else None

    trained_ddpg.save('./trained_models/trained_ddpg.model')

### Model 3: A2C

agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")

if if_using_a2c:
    # set up logger
    tmp_path = RESULTS_DIR + '/a2c'
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_a2c.set_logger(new_logger_a2c)

if exists('./trained_models/trained_a2c.model'):
    trained_a2c = model_a2c.load('./trained_models/trained_a2c.model')
else:
    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000) if if_using_a2c else None

    trained_a2c.save('./trained_models/trained_a2c.model')

### Model 4: TD3

agent = DRLAgent(env=env_train)
TD3_PARAMS = {"batch_size": 100,
              "buffer_size": 1000000,
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

if if_using_td3:
    # set up logger
    tmp_path = RESULTS_DIR + '/td3'
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_td3.set_logger(new_logger_td3)

    if exists('./trained_models/trained_td3.model'):
        trained_td3 = model_td3.load('./trained_models/trained_td3.model')
else:
    trained_td3 = agent.train_model(model=model_td3,
                                    tb_log_name='td3',
                                    total_timesteps=30000) if if_using_td3 else None

    trained_td3.save('./trained_models/trained_td3.model')

### Model 5: SAC

agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

if if_using_sac:
    # set up logger
    tmp_path = RESULTS_DIR + '/sac'
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    # Set new logger
    model_sac.set_logger(new_logger_sac)

if exists('./trained_models/trained_sac.model'):
    trained_sac = model_sac.load('./trained_models/trained_sac.model')
else:
    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=30000) if if_using_sac else None

    trained_sac.save('./trained_models/trained_sac.model')

# ------------------------------
#  Далее торговля
# -----------------------------


