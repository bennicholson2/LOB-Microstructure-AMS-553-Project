# Nicholas Christophides  Nicholas.christophides@stonybrook.edu
# Benjamin Nicholson  Benjamin.nicholson@stonybrook.edu

import simpy
import numpy as np
from simulation_functions import (Distribution, inverse_transform_method_exponential,
                                  output_analysis_data, plot_orderbook_metrics,
                                  multiple_simulations, output_simulation_results,
                                  simulation_results_across_parameters)
from orderbook import OrderBook
from investors import Buyer, Seller


# ----- Establish Market Parameters -----
# View the simulation bid-ask spread evolution
p0 = 100  # set a fundamental price for the market
p0_min = -p0*0.02  # minimum noise
p0_max = p0*0.02  # maximum noise
n_investors = 1000  # this is the types of investors that we create
# if we were to n_investors to 1 then there is only a buyer and a seller who repeatedly return to the queue

buyers_list = []
sellers_list = []


# ----- Initialize Simulation Environment and Orderbook -----
env = simpy.Environment(0)  # initial start time of the simulation
orderbook = OrderBook(p0)  # initialize the orderbook with fundamental price p0


# ----- Specify Arrival Rate Parameters, Price Distribution Parameters, and Create Distributions to Sample From -----
buyers_arrival_rate = 1/n_investors  # the number of buyers per unit time
seller_arrival_rate = 1/n_investors  # the number of sellers per unit time

buyer_arrival_dist = Distribution(
    lambda: inverse_transform_method_exponential(np.random.uniform(0, 1), buyers_arrival_rate))

seller_arrival_dist = Distribution(
    lambda: inverse_transform_method_exponential(np.random.uniform(0, 1), seller_arrival_rate))

price_dist = Distribution(lambda: np.random.uniform(p0_min, p0_max))  # we want the scale to be 2


# ----- Create Holders for Buyers and Sellers -----
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


# ----- Select Length and Run the Simulation -----
hours = 6
minutes = 0
time_elapsed = ((hours*60) + minutes)
# expected time to finish
env.run(until=time_elapsed)


# ----- Collect the Data for Visualization -----
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

# ----- Simulation Results and Visuals -----

simulation_results_neutral = multiple_simulations(30, 100, 0.02, 1,
                                                  1, 6, 0)
simulation_results_bull = multiple_simulations(30, 100, 0.02, 2,
                                               1, 6, 0)
simulation_results_bear = multiple_simulations(30, 100, 0.02, 1,
                                               2, 6, 0)

simulation_results_runs_neutral = output_simulation_results(simulation_results_neutral)[0]
simulation_results_runs_bull = output_simulation_results(simulation_results_bull)[0]
simulation_results_runs_bear = output_simulation_results(simulation_results_bear)[0]

simulation_results_runs_dict = {
    "neutral": simulation_results_runs_neutral,
    "bull": simulation_results_runs_bull,
    "bear": simulation_results_runs_bear
}

overall_results = simulation_results_across_parameters(simulation_results_runs_dict)

print(f'----- Neutral -----\n{overall_results["neutral"]}\n\n----- Bull -----\n{overall_results["bull"]}'
      f'\n\n----- Bear -----\n{overall_results["bear"]}')

(time, best_bids_ts, best_asks_ts, midpoint_ts, spread_ts, completed_wait_times, ongoing_wait_times,
 total_wait_times, bid_queue_size, ask_queue_size, all_bids_prices, all_bids_times, all_asks_prices,
 all_asks_times, all_trades_prices, all_trades_times, orderbook_bids, orderbook_asks) = output_analysis_data(orderbook)

plot_orderbook_metrics(time, best_bids_ts, best_asks_ts, midpoint_ts, spread_ts, completed_wait_times,
                       ongoing_wait_times, bid_queue_size, ask_queue_size, all_bids_prices, all_bids_times,
                       all_asks_prices, all_asks_times,
                       all_trades_prices, all_trades_times, orderbook_bids, orderbook_asks)
