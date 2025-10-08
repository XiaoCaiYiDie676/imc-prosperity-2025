import jsonpickle
from dataclasses import dataclass
from typing import List, Dict

# === Basic Classes ===
import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()



@dataclass
class Order:
    product: str
    price: int
    quantity: int

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

# === Strategy Base Class ===

class BaseStrategy:
    def __init__(self, product: str, params: dict):
        self.product = product
        self.params = params
        self.history = []

    def run(self, state, trader_data: dict) -> List[Order]:
        raise NotImplementedError

# === Fixed Value Strategy (for RAINFOREST_RESIN) ===

class FixedValueStrategy(BaseStrategy):
    def run(self, state, trader_data):
        order_depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        orders = []

        fv = self.params["fair_value"]
        max_pos = self.params["position_limit"]
        lot_size = self.params.get("lot_size", 5)
        inner_width = self.params["inner_width"]  # tight spread
        outer_width = self.params["outer_width"]  # wider spread

        bid_levels = [fv - inner_width, fv - outer_width]
        ask_levels = [fv + inner_width, fv + outer_width]

        # --- Buy below FV ---
        for price in bid_levels:
            for ask_price, volume in sorted(order_depth.sell_orders.items()):
                if ask_price <= price and position < max_pos:
                    qty = min(volume, max_pos - position, lot_size)
                    if qty > 0:
                        orders.append(Order(self.product, ask_price, qty))
                        position += qty
                        break  # only one price level per tick for simplicity

        # --- Sell above FV ---
        for price in ask_levels:
            for bid_price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= price and position > -max_pos:
                    qty = min(volume, max_pos + position, lot_size)
                    if qty > 0:
                        orders.append(Order(self.product, bid_price, -qty))
                        position -= qty
                        break

        return orders
# === Mean Reverting Strategy (for KELP) ===

class KelpStrategy(BaseStrategy):
    def run(self, state, trader_data):
        order_depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        orders = []

        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return orders

        self.history.append(mid_price)
        window = self.params["window"]
        max_pos = self.params["position_limit"]
        lot_size = self.params["lot_size"]
        inner_width = self.params["inner_width"]
        outer_width = self.params["outer_width"]

        avg_price = sum(self.history[-window:]) / min(len(self.history), window)

        bid_levels = [avg_price - inner_width, avg_price - outer_width]
        ask_levels = [avg_price + inner_width, avg_price + outer_width]

        # Buy below average
        for price in bid_levels:
            for ask_price, volume in sorted(order_depth.sell_orders.items()):
                if ask_price <= price and position < max_pos:
                    qty = min(volume, max_pos - position, lot_size)
                    if qty > 0:
                        orders.append(Order(self.product, ask_price, qty))
                        position += qty
                        break

        # Sell above average
        for price in ask_levels:
            for bid_price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= price and position > -max_pos:
                    qty = min(volume, max_pos + position, lot_size)
                    if qty > 0:
                        orders.append(Order(self.product, bid_price, -qty))
                        position -= qty
                        break

        return orders

    def get_mid_price(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if not bids or not asks:
            return None
        return (max(bids.keys()) + min(asks.keys())) / 2
    
# === Z-Score Mean Reverting Strategy (for SQUID_INK) ===

class SquidInkStrategy(BaseStrategy):
    def run(self, state, trader_data):
        order_depth = state.order_depths[self.product]
        position = state.position.get(self.product, 0)
        orders = []

        mid_price = self.get_mid_price(order_depth)
        if mid_price is None:
            return orders

        self.history.append(mid_price)
        max_pos = self.params["position_limit"]
        window = self.params["window"]
        threshold = self.params["zscore_threshold"]
        inner_width = self.params["inner_width"]
        outer_width = self.params["outer_width"]
        lot_size = self.params["lot_size"]

        if len(self.history) < window:
            return orders

        avg = sum(self.history[-window:]) / window
        std = (sum((x - avg) ** 2 for x in self.history[-window:]) / window) ** 0.5
        zscore = (mid_price - avg) / std if std > 0 else 0

        # Place layered orders in the opposite direction of extreme swings
        if zscore < -threshold:
            # Buy small chunks below mid
            for offset in [outer_width, inner_width]:
                price = int(mid_price - offset)
                for ask_price, volume in sorted(order_depth.sell_orders.items()):
                    if ask_price <= price and position < max_pos:
                        qty = min(volume, max_pos - position, lot_size)
                        if qty > 0:
                            orders.append(Order(self.product, ask_price, qty))
                            position += qty
                            break

        elif zscore > threshold:
            # Sell small chunks above mid
            for offset in [outer_width, inner_width]:
                price = int(mid_price + offset)
                for bid_price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    if bid_price >= price and position > -max_pos:
                        qty = min(volume, max_pos + position, lot_size)
                        if qty > 0:
                            orders.append(Order(self.product, bid_price, -qty))
                            position -= qty
                            break

        return orders

    def get_mid_price(self, order_depth):
        bids = order_depth.buy_orders
        asks = order_depth.sell_orders
        if not bids or not asks:
            return None
        return (max(bids.keys()) + min(asks.keys())) / 2
    
# === Parameters ===

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "inner_width": 2,
        "outer_width": 5,
        "position_limit": 50,
        "lot_size": 5
    },
    Product.KELP: {
        "window": 10,
        "inner_width": 3,
        "outer_width": 6,
        "position_limit": 50,
        "lot_size": 5
    },
    Product.SQUID_INK: {
        "window": 10,
        "zscore_threshold": 1.2,
        "take_width": 4,
        "position_limit": 50,
        "lot_size": 5
    },
}

# === Trader Class ===

class Trader:
    def __init__(self):
        self.strategies = {
            Product.RAINFOREST_RESIN: FixedValueStrategy(Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN]),
            Product.KELP: KelpStrategy(Product.KELP, PARAMS[Product.KELP]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, PARAMS[Product.SQUID_INK])
        }
        self.trader_data = {}

    def run(self, state):
        if state.traderData:
            self.trader_data = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        for product, strategy in self.strategies.items():
            result[product] = strategy.run(state, self.trader_data)

        trader_data = jsonpickle.encode(self.trader_data)
        logger.flush(state, result, conversions, trader_data)

        return result, conversions, trader_data