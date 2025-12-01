# Nicholas Christophides  Nicholas.christophides@stonybrook.edu
# Benjamin Nicholson  Benjamin.nicholson@stonybrook.edu

import simpy
import numpy as np
from simulation_functions import Distribution, inverse_transform_method_exponential
from orderbook import OrderBook
from investors import Buyer, Seller
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(1127)


# ----- Establish Market Parameters -----
# View the simulation bid-ask spread evolution
p0 = 100  # set a fundamental price for the market
p0_min = -p0*0.02  # minimum noise
p0_max = p0*0.02  # maximum noise
n_investors = 1000  # this is the types of investors that we create
# if we were to n_investors to 1 then there is only a buyer and a seller who repeatedly return to the queue
buyers_arrival_rate = 1/n_investors  # the number of buyers per unit time
seller_arrival_rate = 1/n_investors  # the number of sellers per unit time

buyers_list = []
sellers_list = []

env = simpy.Environment(0)  # initial start time of the simulation
orderbook = OrderBook(p0)  # initialize the orderbook with fundamental price p0

buyer_arrival_dist = Distribution(
    lambda: inverse_transform_method_exponential(np.random.uniform(0, 1), buyers_arrival_rate))

seller_arrival_dist = Distribution(
    lambda: inverse_transform_method_exponential(np.random.uniform(0, 1), seller_arrival_rate))

price_dist = Distribution(lambda: np.random.uniform(p0_min, p0_max))  # we want the scale to be 2

for i in range(n_investors):  # generate n_investors buyers with their own count
    buyer = Buyer(f'buy_{i}', price_dist, buyer_arrival_dist)
    buyers_list.append(buyer)

for i in range(n_investors):  # generate n_investors sellers with their own count
    seller = Seller(f'sell{i}', price_dist, seller_arrival_dist)
    sellers_list.append(seller)

for b in buyers_list:
    env.process(b.run(env, orderbook))
for s in sellers_list:
    env.process(s.run(env, orderbook))

hours = 6
minutes = 0
time_elapsed = ((hours*60) + minutes)
# expected time to finish
env.run(until=time_elapsed)


all_bids_price = []
all_bids_time = []
for i in range(len(orderbook.all_bids)):
    all_bids_price.append(orderbook.all_bids[i][0])
    all_bids_time.append(orderbook.all_bids[i][1])

all_asks_price = []
all_asks_time = []
for i in range(len(orderbook.all_asks)):
    all_asks_price.append(orderbook.all_asks[i][0])
    all_asks_time.append(orderbook.all_asks[i][1])

all_trades_price = []
all_trades_time = []
for i in range(len(orderbook.trade_history)):
    all_trades_time.append(orderbook.trade_history[i][1])
    all_trades_price.append(orderbook.trade_history[i][0])

plt.plot(all_bids_time, all_bids_price, label='All Bids', color='Green')
plt.plot(all_asks_time, all_asks_price, label='All Asks', color='Red')
plt.scatter(all_trades_time, all_trades_price, label='Trades', color='Orange')
plt.title('Bids, Asks and Trades Through Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

orderbook_history = pd.DataFrame(orderbook.orderbook_history)

plt.plot(orderbook_history['time'][100:200], orderbook_history['best_bid'][100:200], label="Best Bid")
plt.plot(orderbook_history['time'][100:200], orderbook_history['best_ask'][100:200], label="Best Ask")
# plt.plot(orderbook_history['time'],orderbook_history['spread'])
plt.plot(orderbook_history['time'][100:200], orderbook_history['midpoint'][100:200], label="Midpoint")
plt.legend()
plt.title("Bid-Ask Spread over Time")
plt.show()
