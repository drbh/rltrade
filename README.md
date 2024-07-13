# RLTRADE

This repo is a tiny exploration of RL models applied to stock trading.

This project implements a simple actor-critic lstm model that takes an input of various indicators and outputs a buy/sell signal (-1 to 1).

## how to use

Train and run the model. This will save the model to disk and then run it on the latest data.

```bash
python train.py --train --epochs 400
```

Run it without training

```bash
python train.py
```

Example of trades
![rltrade1](https://github.com/user-attachments/assets/f1eea003-017f-4e18-807f-9b8d7e63b81f)


## are we rich yet?

nope not yet, this is 100% just a research project and do not recommend using this for any real world purposes. This is simply a toy example outlining how market data can be used to train a model to make buy/sell decisions.

## setup

get your virtual environment up and running with the following commands:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## notes / ideas

- [ ] Add more indicators and possibly ingest news headlines, embed them and feed them into the model
- [ ] Focus more on portfolio management and risk assessment. Currently the model doesnt take into account it's current position and the risk associated with it, this is often the better/more holistic approach to take when trading.
- [ ] Add an "inferece" mode that basically just runs the model on the latest data and outputs a buy/sell signal, then implement a small paper trading bot that runs daily.
- [ ] It would be cool to move to an online approach and train on much more granular data, like minute by minute data and run it in real time... but that's a bit out of scope for this project.
