# Nicholas Christophides  Nicholas.christophides@stonybrook.edu
# Benjamin Nicholson  Benjamin.nicholson@stonybrook.edu


import numpy as np
import matplotlib.pyplot as plt
import simpy
from orderbook import OrderBook
from investors import Buyer, Seller
import pandas as pd
import scipy.stats as st


class Distribution:
    """
    Distributions class will be used for random variate generation
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self):
        return self.sampler()


def inverse_transform_method_exponential(u, arrival_rate):
    return - (1 / arrival_rate) * np.log(u)


def output_analysis_data(orderbook):
    # --- Time series snapshots at each event ---
    time = []
    best_bids_ts = []
    best_asks_ts = []
    midpoint_ts = []
    spread_ts = []
    completed_wait_times = []
    ongoing_wait_times = []
    total_wait_times = []
    bid_queue_size = []
    ask_queue_size = []
    orderbook_bids = []
    orderbook_asks = []

    for snapshot in orderbook.orderbook_history:
        time.append(snapshot['time'])
        best_bids_ts.append(snapshot['best_bid'])
        best_asks_ts.append(snapshot['best_ask'])
        midpoint_ts.append(snapshot['midpoint'])
        spread_ts.append(snapshot['spread'])

        cw = snapshot['completed_wait_times']
        ow = snapshot['ongoing_wait_times']
        tw = snapshot['total_wait_times']

        completed_wait_times.append(np.mean(cw) if len(cw) > 0 else np.nan)
        ongoing_wait_times.append(np.mean(ow) if len(ow) > 0 else np.nan)
        total_wait_times.append(np.mean(tw) if len(tw) > 0 else np.nan)

        bid_queue_size.append(snapshot['bid_queue_size'])
        ask_queue_size.append(snapshot['ask_queue_size'])

    # --- Full event-level bid/ask history ---
    all_bids_prices = [b[0] for b in orderbook.all_bids]
    all_bids_times = [b[1] for b in orderbook.all_bids]

    all_asks_prices = [a[0] for a in orderbook.all_asks]
    all_asks_times = [a[1] for a in orderbook.all_asks]

    # --- Trades ---
    all_trades_prices = [t[0] for t in orderbook.trade_history]
    all_trades_times = [t[1] for t in orderbook.trade_history]

    # end of simulation order book
    for bids in orderbook.bids:
        orderbook_bids.append(-bids[0])
    for asks in orderbook.asks:
        orderbook_asks.append(asks[0])

    return (
        time,
        best_bids_ts, best_asks_ts, midpoint_ts, spread_ts,
        completed_wait_times, ongoing_wait_times, total_wait_times,
        bid_queue_size, ask_queue_size,
        all_bids_prices, all_bids_times,
        all_asks_prices, all_asks_times,
        all_trades_prices, all_trades_times,
        orderbook_bids, orderbook_asks
    )


def plot_orderbook_metrics(
        time, best_bids, best_asks, midpoint,
        spread, comp_wait, ong_wait,
        bqsize, aqsize,
        bids_price, bids_time,
        asks_price, asks_time,
        trades_prices, trades_times,
        end_bids, end_asks):

    fig = plt.figure(figsize=(10, 12))

    # 4 rows, 2 columns → last row full width
    gs = fig.add_gridspec(4, 2)

    # PANEL 1 ‒ Best Bid / Ask / Mid
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, best_bids, label="Best Bid")
    ax1.plot(time, best_asks, label="Best Ask")
    ax1.plot(time, midpoint, label="Midpoint", linestyle="--")
    ax1.set_title("Best Bid / Best Ask / Midpoint")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # PANEL 2 ‒ Completed wait
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, comp_wait)
    ax2.set_title("Completed Orders – Avg Wait Time")
    ax2.grid(alpha=0.3)

    # PANEL 3 ‒ Ongoing wait
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time, ong_wait)
    ax3.set_title("Ongoing Orders – Avg Queue Wait Time")
    ax3.grid(alpha=0.3)

    # PANEL 4 ‒ Queue sizes
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time, bqsize, label="Bid Queue Size")
    ax4.plot(time, aqsize, label="Ask Queue Size")
    ax4.set_title("Queue Size")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # PANEL 5 ‒ All Bids/Asks/Trades
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(bids_time, bids_price, color="green", label="All Bids")
    ax5.plot(asks_time, asks_price, color="red", label="All Asks")
    ax5.scatter(trades_times, trades_prices, s=10, color="orange", label="Trades")
    ax5.set_title("Bids, Asks, and Trades Through Time")
    ax5.legend()
    ax5.grid(alpha=0.3)

    # PANEL 6 is the Spread — if you want it kept
    # If not, you can comment this out
    ax6 = fig.add_subplot(gs[0, 1])
    ax6.plot(time, spread)
    ax6.set_title("Spread")
    ax6.grid(alpha=0.3)

    # PANEL 7 (FINAL) ‒ Big histogram across whole width
    ax7 = fig.add_subplot(gs[3, :])
    ax7.hist(end_bids, bins=20, alpha=0.6, label="Final Bids", color="green")
    ax7.hist(end_asks, bins=20, alpha=0.6, label="Final Asks", color="red")
    ax7.set_title("Final Limit Order Book Depth (Histogram)")
    ax7.set_xlabel("Price")
    ax7.set_ylabel("Frequency")
    ax7.legend()
    ax7.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def multiple_simulations(n_sims, p0, noise_lvl, buyer_arrival_rate, seller_arrival_rate, hours, minutes):
    """
    P0: Initial Price.
    noise_lvl: Pull from uniform distribution for noise using ratio difference of p0 price
    n_investor_types: Get the number of types of distributions each with their own interarrival distributions and price
    distributions
    """

    simulation_runs = {}

    p0 = p0
    p0_min = -p0 * noise_lvl
    p0_max = p0 * noise_lvl

    for i in range(n_sims):
        np.random.seed(i)

        buyer_arrival_dist = Distribution(
            lambda: np.random.exponential(1 / buyer_arrival_rate))  # we can adjust the arrival rate for the buyers
        seller_arrival_dist = Distribution(
            lambda: np.random.exponential(1 / seller_arrival_rate))  # we can adjust the arrival rate for the sellers

        buyer_price_dist_noise = Distribution(
            lambda: np.random.uniform(p0_min, p0_max))  # we can adjust the noise for the buyers
        seller_price_dist_noise = Distribution(
            lambda: np.random.uniform(p0_min, p0_max))  # we can adjust the noise for the sellers

        env = simpy.Environment(0)
        orderbook = OrderBook(p0)

        buyer = Buyer('Buyer', buyer_price_dist_noise, buyer_arrival_dist)
        seller = Seller('Seller', seller_price_dist_noise, seller_arrival_dist)

        env.process(buyer.run(env, orderbook))
        env.process(seller.run(env, orderbook))

        time_elapsed = ((hours * 60) + minutes)

        env.run(until=time_elapsed)

        (time, best_bids_ts, best_asks_ts, midpoint_ts, spread_ts, completed_wait_times, ongoing_wait_times,
         total_wait_times, bid_queue_size, ask_queue_size, all_bids_prices, all_bids_times, all_asks_prices,
         all_asks_times, all_trades_prices, all_trades_times,
         orderbook_bids, orderbook_asks) = (output_analysis_data(orderbook))

        results = pd.DataFrame({
            "time": time,
            "best_bids": best_bids_ts,
            "best_asks": best_asks_ts,
            "midpoint": midpoint_ts,
            "spread": spread_ts,
            "completed_wait_times": completed_wait_times,
            "ongoing_wait_times": ongoing_wait_times,
            "total_wait_times": total_wait_times,
            "bid_queue_size": bid_queue_size,
            "ask_queue_size": ask_queue_size,
        }, index=time)

        extra = {
            "all_bids_prices": all_bids_prices,
            "all_bids_times": all_bids_times,
            "all_asks_prices": all_asks_prices,
            "all_asks_times": all_asks_times,
            "all_trades_prices": all_trades_prices,
            "all_trades_times": all_trades_times,
            "final_orderbook_bids": orderbook_bids,
            "final_orderbook_asks": orderbook_asks,
            "order filled": orderbook.pct_filled()
        }

        simulation_runs[str(i)] = {
            "timeseries": results,
            "extra": extra
        }

    return simulation_runs


def output_simulation_results(simulation_runs):
    best_bids = []
    best_asks = []
    midpoints = []
    spreads = []
    completed_wt = []
    ongoing_wt = []
    total_wt = []
    bid_q = []
    ask_q = []
    pct_filled = []
    final_bids = []
    final_asks = []

    for i in range(len(simulation_runs)):
        df = simulation_runs[str(i)]['timeseries']
        df2 = simulation_runs[str(i)]['extra']

        best_bids.append(df['best_bids'].mean())
        best_asks.append(df['best_asks'].mean())
        midpoints.append(df['midpoint'].mean())
        spreads.append(df['spread'].mean())
        completed_wt.append(df['completed_wait_times'].mean())
        ongoing_wt.append(df['ongoing_wait_times'].mean())
        total_wt.append(df['total_wait_times'].mean())
        bid_q.append(df['bid_queue_size'].mean())
        ask_q.append(df['ask_queue_size'].mean())
        pct_filled.append(df2['order filled'])
        final_bids.append(df2['final_orderbook_bids'])
        final_asks.append(df2['final_orderbook_asks'])

    # build final summary table
    summary = pd.DataFrame({
        "best_bids": best_bids,
        "best_asks": best_asks,
        "midpoints": midpoints,
        "spreads": spreads,
        "completed_wait_times": completed_wt,
        "ongoing_wait_times": ongoing_wt,
        "total_wait_times": total_wt,
        "bid_queue_size": bid_q,
        "ask_queue_size": ask_q,
        "pct_filled": pct_filled,
    })

    summary_values = summary.mean()

    return summary, summary_values, final_bids, final_asks


def simulation_results_across_parameters(sim_results_dict):
    """
    sim_results_dict:
        key: parameter label
        value: summary DataFrame containing metrics across replications
    """

    markets = list(sim_results_dict.keys())
    metrics = sim_results_dict[markets[0]].columns

    ci_df = pd.DataFrame(index=metrics, columns=markets)

    for market in markets:
        summary_df = sim_results_dict[market]

        for metric in metrics:
            data = summary_df[metric]  # vector across replications
            ci_tuple = confidence_intervals(data)
            ci_df.loc[metric, market] = ci_tuple

    return ci_df


def confidence_intervals(data):
    mean = np.mean(data)
    standard_error = st.sem(data)
    ci = tuple(st.t.interval(0.95, len(data) - 1, loc=mean, scale=standard_error))
    return ci
