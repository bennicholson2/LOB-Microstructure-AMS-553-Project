# Nicholas Christophides  Nicholas.christophides@stonybrook.edu
# Benjamin Nicholson  Benjamin.nicholson@stonybrook.edu


from orderbook import Order


class Investor:
    """
    Attributes:
    - id: Unique identifier for the investor
    - price_dist: Distribution object for price noise
    - arrival_dist: Distribution object for inter-arrival times
    Methods:
    - get_valuation(orderbook): returns the valuation of the investor based on the order book
    - map_price(val, noise): maps the valuation and noise to a price (to be implemented in subclasses)
    - generate_price(orderbook): generates a price based on valuation and noise

    Overview:
    An investor is a parent class to buyers and sellers.
    Each investor comes to the queue with their ID (used as an identifier), a price and an arrival time.
    """

    def __init__(self, id, price_dist, arrival_dist):  # investors have an id, price, and interarrival time
        self.id = id
        self.price_dist = price_dist  # this price distribution is the level of noise (or belief) relative to the fundamental price
        self.arrival_dist = arrival_dist

    def get_valuation(self,
                      orderbook):  # get the fundamental price through the method from the orderbook get midpoint price
        return (
                           orderbook.best_ask() + orderbook.best_bid()) / 2 \
            if orderbook.best_bid() is not None and orderbook.best_ask() is not None else orderbook.p0

    def map_price(self, val, noise):
        raise NotImplementedError

    def generate_price(self, orderbook):  # generate the price by the fundamental price and the noise
        noise = self.price_dist.sample()  # sample from the noise price distribution
        val = self.get_valuation(orderbook)  # get the value of the fundamental price
        return round(self.map_price(val, noise), 2)


class Buyer(Investor):
    def map_price(self, val, noise):  # each buyer is an investor type
        return val + noise

    def run(self, env, orderbook):
        while True:  # while there are events in the buyer list
            yield env.timeout(
                self.arrival_dist.sample())  # wait until a sampled arrival distribution has taken place then proceed to have orders go through the simulaiton
            price = self.generate_price(orderbook)  # generate a price
            order_id = orderbook.next_order_id()  # create the order id
            order = Order(order_id, self.id, price, env.now, "buy")  # add the order to the order book
            orderbook.add_order(order)


class Seller(Investor):
    def map_price(self, val, noise):
        return val + noise

    def run(self, env, orderbook):
        while True:
            yield env.timeout(
                self.arrival_dist.sample())  # wait until a sampled arrival distribution has taken place then proceed to have orders go through the simulation
            price = self.generate_price(orderbook)  # generate a price
            order_id = orderbook.next_order_id()  # create the order id
            order = Order(order_id, self.id, price, env.now, "sell")  # add the order to the order book
            orderbook.add_order(order)


class Buyer(Investor):
    def map_price(self, val, noise):  # each buyer is an investor type
        return val + noise

    def run(self, env, orderbook):
        while True:  # while there are events in the buyer list
            # wait until a sampled arrival distribution has taken place
            # then proceed to have orders go through the simulation
            yield env.timeout(self.arrival_dist.sample())
            price = self.generate_price(orderbook)  # generate a price
            order_id = orderbook.next_order_id()  # create the order id
            order = Order(order_id, self.id, price, env.now, "buy")  # add the order to the order book
            orderbook.add_order(order)


class Seller(Investor):
    def map_price(self, val, noise):
        return val + noise

    def run(self, env, orderbook):
        while True:
            # wait until a sampled arrival distribution has taken place
            # then proceed to have orders go through the simulation
            yield env.timeout(self.arrival_dist.sample())
            price = self.generate_price(orderbook)  # generate a price
            order_id = orderbook.next_order_id()  # create the order id
            order = Order(order_id, self.id, price, env.now, "sell")  # add the order to the order book
            orderbook.add_order(order)
