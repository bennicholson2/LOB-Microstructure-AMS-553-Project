# Nicholas Christophides  Nicholas.christophides@stonybrook.edu
# Benjamin Nicholson  Benjamin.nicholson@stonybrook.edu

import heapq


class OrderBook:
    """
    Attributes:
    - p0: initial price of the asset
    - bids: list of buy orders (max heap)
    - asks: list of sell orders (min heap)

    Methods:
    - best_bid(): returns the highest bid price
    - best_ask(): returns the lowest ask price
    - midpoint_price(): returns the midpoint price between best bid and best ask
    - add_order(order): adds an order to the order book
    """

    def __init__(self, p0):
        self.p0 = p0  # initial fundamental price of a product
        self.bids = []  # max heap - priority queue
        self.asks = []  # min heap - priority queue of asks
        self.best_bid_list = []
        self.best_ask_list = []
        self.all_bids = []
        self.all_asks = []
        self.trade_history = []
        self.next_order_id_counter = 0
        self.orderbook_history = []
        self.all_orders = []
        self.pct_filled_ts = []

    """
    The following functions are made to keep track of how the orderbook progresses with time
    """

    def best_bid(self):  # simply look at the minimum of the negative (largest prices of the bid queue)
        current_best_bid = -self.bids[0][0] if self.bids else None
        self.best_bid_list.append(current_best_bid)
        return current_best_bid

    def best_ask(self):  # look at the minimum of the seller ask prices
        current_best_ask = self.asks[0][0] if self.asks else None
        self.best_ask_list.append(current_best_ask)
        return current_best_ask

    def compute_wait_times(self, current_time):  # we want to compute wait times
        completed_wait_times = []
        current_wait_times = []
        total_wait_time = []
        for o in self.all_orders:
            if o.is_filled:
                completed_wait_times.append(o.execution_time - o.time)
            else:
                current_wait_times.append(current_time - o.time)
            total_wait_time.append(current_time - o.time if not o.is_filled else o.execution_time - o.time)
        return completed_wait_times, current_wait_times, total_wait_time

    def record_state(self, current_time):
        completed_wait_times, ongoing_wait_times, total_wait_time = self.compute_wait_times(current_time)
        bid = self.best_bid()
        ask = self.best_ask()
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        spread = ask - bid if bid is not None and ask is not None else None

        snapshot = {
            "time": current_time,
            "best_bid": bid,
            "best_ask": ask,
            "midpoint": mid,
            "spread": spread,
            "completed_wait_times": completed_wait_times,
            "ongoing_wait_times": ongoing_wait_times,
            "total_wait_times": total_wait_time,
            "bid_queue_size": len(self.bids),
            "ask_queue_size": len(self.asks),
        }
        self.orderbook_history.append(snapshot)

    def add_order(self, order):  # add order to the limit order book with the information from the 'order' class
        if order.side == "buy":
            self._process_buy(order)
        else:
            self._process_sell(order)
        self.all_orders.append(order)
        self.record_state(
            order.time)  # record the state at the end of each order added as this is the only time that changes happen to the orderbook

    def _process_buy(self, order):  # buy agents actions
        self.all_bids.append((order.price, order.time))
        while self.asks and (
                order.price >= self.best_ask()):  # assuming that asks exist and the price is greater than the best ask then execute trade
            best_ask_price, time, oid, ask_order = heapq.heappop(
                self.asks)  # we take the highest priority from the ask and assume there trade has been matched by the buyers

            ask_order.is_filled = True  # the ask order is the one that is sitting in the queue
            ask_order.execution_time = order.time

            order.is_filled = True  # the order is the new order which matches the ask order
            order.execution_time = order.time

            self.trade_history.append((best_ask_price, time, ask_order))  # add this trade to the trade history
            return

            # if there is no match then push the order to the bids
        heapq.heappush(self.bids, (-order.price, order.time, order.id, order))

    def _process_sell(self, order):
        self.all_asks.append((order.price, order.time))
        while self.bids and (
                order.price <= self.best_bid()):  # assuming that bids exist and the price is less than the best bid then execute trade
            best_bid_price, time, oid, bid_order = heapq.heappop(
                self.bids)  # we take the highest priority from the bid and assume there trade has been matched by the sellers
            best_bid_price = -best_bid_price  # take the negative for the actual price due to heapq properties finding minimum
            bid_order.is_filled = True
            bid_order.execution_time = order.time

            order.is_filled = True
            order.execution_time = order.time
            self.trade_history.append((best_bid_price, time, bid_order))  # add to trade history
            return
        # if there is no match then the order to the asks pile
        heapq.heappush(self.asks, (order.price, order.time, order.id, order))

    def pct_filled(self):
        filled = sum(o.is_filled for o in self.all_orders)
        return filled / len(self.all_orders)

    def next_order_id(self):  # keep a counter of the number of orders that have entered the order book
        self.next_order_id_counter += 1
        return self.next_order_id_counter


class Order:
    def __init__(self, order_id, investor_id, price, time, side):
        """
        Attributes:
        investor_id: Unique identifier for the investor placing the order
        price: The price at which the order is placed
        time: The timestamp when the order is placed
        side: 'buy' or 'sell' indicating the type of order

        """
        self.id = order_id
        self.investor_id = investor_id
        self.price = price
        self.time = time
        self.side = side

        self.execution_time = None
        self.is_filled = False
