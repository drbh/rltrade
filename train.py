from datetime import datetime, timedelta
import argparse

# data manipulation
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# modeling/learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file
from safetensors import safe_open


# for the data
import yfinance as yf

# plotting at the end
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# our model/parameters imported
from model import ActorCritic

from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    create_sequences,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 10
MAX_TRADES = 40
SEQ_LENGTH = 20
FUTURE_WINDOW = 10
CLIP_EPSILON = 0.2
VALUE_COEFF = 0.5
ENTROPY_COEFF = 0.01


def calculate_indicators(data):
    df = data.copy()
    df["Returns"] = df["Close"].pct_change()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility"] = df["Returns"].rolling(window=20).std()
    df["RSI"] = calculate_rsi(df["Close"])
    df["MACD"] = calculate_macd(df["Close"])
    df["ATR"] = calculate_atr(df)
    df["FutureReturn"] = (
        df["Close"].pct_change(periods=FUTURE_WINDOW).shift(-FUTURE_WINDOW)
    )
    return df.dropna()


def preprocess_data(data):
    df = calculate_indicators(data)
    features = ["Close", "Returns", "MA_5", "MA_20", "Volatility", "RSI", "MACD", "ATR"]
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler, features


def ppo_update(
    model, optimizer, states, old_log_probs, returns, advantages, clip_epsilon
):
    for _ in range(5):
        new_actions, new_log_probs, values = model.get_action(states)
        ratio = (new_log_probs - old_log_probs.detach()).exp()

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = ((values - returns) ** 2).mean()

        entropy = -new_log_probs.mean()

        loss = actor_loss + VALUE_COEFF * value_loss - ENTROPY_COEFF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

    return loss.item()


def train_ppo(model, data, features, epochs):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    sequences, targets = create_sequences(data, SEQ_LENGTH, features)
    dataset = TensorDataset(torch.FloatTensor(sequences), torch.FloatTensor(targets))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for states, returns in dataloader:
            actions, old_log_probs, values = model.get_action(states)
            advantages = returns.unsqueeze(1) - values.detach()

            loss = ppo_update(
                model,
                optimizer,
                states,
                old_log_probs,
                returns.unsqueeze(1),
                advantages,
                CLIP_EPSILON,
            )
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} completed, Avg Loss: {avg_loss:.4f}")


## Simple trading strategy (issue: this executes too fast)
# def evaluate_model(model, test_data, features, scaler):
#     model.eval()
#     test_sequences, _ = create_sequences(test_data, SEQ_LENGTH, features)
#     test_sequences = torch.FloatTensor(test_sequences)

#     trades = []
#     with torch.no_grad():
#         for i in range(len(test_sequences)):
#             action, _, _ = model.get_action(test_sequences[i].unsqueeze(0))
#             if action.item() > 0.5 and len(trades) < MAX_TRADES:
#                 entry_price = test_data.iloc[i + SEQ_LENGTH - 1]["Close"]
#                 entry_date = test_data.index[i + SEQ_LENGTH - 1]
#                 future_prices = test_data.iloc[
#                     i + SEQ_LENGTH - 1 : i + SEQ_LENGTH - 1 + FUTURE_WINDOW
#                 ]["Close"]
#                 exit_price = future_prices.max()
#                 exit_date = future_prices.idxmax()

#                 # scaled prices
#                 trades.append((entry_date, exit_date, entry_price, exit_price))

#     return trades


def evaluate_model(
    model, test_data, features, scaler, signal_threshold, profit_target, stop_loss
):
    model.eval()
    test_sequences, _ = create_sequences(test_data, SEQ_LENGTH, features)
    test_sequences = torch.FloatTensor(test_sequences)

    trades = []
    current_position = None
    consecutive_signals = 0

    with torch.no_grad():
        for i in range(len(test_sequences)):
            action, _, _ = model.get_action(test_sequences[i].unsqueeze(0))
            current_price = test_data.iloc[i + SEQ_LENGTH - 1]["Close"]
            current_date = test_data.index[i + SEQ_LENGTH - 1]

            if action.item() > 0.5:
                consecutive_signals += 1
            else:
                consecutive_signals = 0

            if current_position is None and consecutive_signals >= signal_threshold:
                # open new position
                current_position = (current_date, current_price)
                consecutive_signals = 0
            elif current_position is not None:
                entry_date, entry_price = current_position
                returns = (current_price - entry_price) / entry_price

                if (
                    returns >= profit_target
                    or returns <= stop_loss
                    or (
                        action.item() <= 0.5 and consecutive_signals >= signal_threshold
                    )
                ):
                    # close position
                    trades.append(
                        (entry_date, current_date, entry_price, current_price)
                    )
                    current_position = None
                    consecutive_signals = 0

                    if len(trades) >= MAX_TRADES:
                        break

    return trades


def plot_trading_signals(data, trades, title="Trading Signals"):
    fig, ax = plt.subplots(figsize=(15, 8))

    # closing price
    ax.plot(data.index, data["Close"], label="Close Price", color="blue")

    # buy signals
    buy_dates = [trade[0] for trade in trades]
    buy_prices = [data.loc[date, "Close"] for date in buy_dates]
    ax.scatter(
        buy_dates, buy_prices, color="green", label="Buy Signal", marker="^", s=100
    )

    # sell signals
    sell_dates = [trade[1] for trade in trades]
    sell_prices = [data.loc[date, "Close"] for date in sell_dates]
    ax.scatter(
        sell_dates, sell_prices, color="red", label="Sell Signal", marker="v", s=100
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    # format x-axis to display dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ticker", type=str, default="^GSPC")
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--epochs", type=int, default=EPOCHS)
    args = argparser.parse_args()

    ticker = args.ticker  # S&P 500
    should_train = args.train
    EPOCHS = args.epochs  # override default epochs if provided

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    data = yf.download(ticker, start=start_date, end=end_date)

    # prep data
    processed_data, scaler, features = preprocess_data(data)

    # split train and test
    train_data = processed_data.iloc[:-500]
    test_data = processed_data.iloc[-500:]

    model = ActorCritic(len(features))

    if should_train:
        # run training
        train_ppo(model, train_data, features, EPOCHS)

        # save the trained model
        save_file(model.state_dict(), "ppo_trading_model.safetensors")
    else:
        # load the trained model
        tensors = {}
        with safe_open("ppo_trading_model.safetensors", framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        model.load_state_dict(tensors)
        model.eval()

    # evaluate the model
    signal_threshold = 4  # num consecutive signals required to open a position
    profit_target = 0.1  # profit target
    stop_loss = -0.9  # stop loss

    trades = evaluate_model(
        model, test_data, features, scaler, signal_threshold, profit_target, stop_loss
    )

    # un scale prices to calculate returns
    close_scaler = scaler.scale_[features.index("Close")]
    close_min = scaler.data_min_[features.index("Close")]

    def unscale_price(scaled_price):
        return (scaled_price / close_scaler) + close_min

    model_returns = [
        (unscale_price(trade[3]) / unscale_price(trade[2]) - 1) for trade in trades
    ]

    model_return = np.prod([1 + ret for ret in model_returns]) - 1

    # calculate buy-and-hold return
    buy_and_hold_return = (data["Close"].iloc[-1] - data["Close"].iloc[-500]) / data[
        "Close"
    ].iloc[-500]

    # calculate Sharpe ratio and max drawdown
    model_sharpe = calculate_sharpe_ratio(pd.Series(model_returns))
    model_max_drawdown = calculate_max_drawdown(pd.Series(model_returns))

    print(f"Buy-and-hold return: {buy_and_hold_return:.4f}")
    print(f"Model return (best {len(trades)} trades): {model_return:.4f}")
    print(f"Number of trades: {len(trades)}")
    print(f"Sharpe ratio: {model_sharpe:.4f}")
    print(f"Max drawdown: {model_max_drawdown:.4f}")

    # print trade details
    for i, trade in enumerate(trades, 1):
        unscaled_entry = unscale_price(trade[2])
        unscaled_exit = unscale_price(trade[3])
        print(f"Trade {i}:")
        print(f"  Entry: {trade[0].date()} at {unscaled_entry:.2f}")
        print(f"  Exit:  {trade[1].date()} at {unscaled_exit:.2f}")
        print(f"  Return: {(unscaled_exit/unscaled_entry - 1)*100:.2f}%")
        print()

    #
    plot_trading_signals(data.iloc[-500:], trades, title=f"{ticker} Trading Signals")
