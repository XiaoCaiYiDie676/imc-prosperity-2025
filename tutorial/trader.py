from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

import jsonpickle
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

from typing import List, Dict, Deque
from collections import deque
import json
import jsonpickle
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

@dataclass
class StrategyParams:
    take_width: float
    clear_width: float
    adverse_volume: int = 0
    reversion_beta: float = 0.2
    disregard_edge: float = 1
    join_edge: float = 2
    default_edge: float = 4
    position_limit: int = 50
    soft_position_limit: int = 40
    window_size: int = 10
    fair_value: float = None
    ema_alpha: float = 0.2  # For exponential moving average
    gamma: float = 0.1      # For GLFT strategy
    order_amount: int = 20  # For GLFT strategy

PARAMS = {
    Product.RAINFOREST_RESIN: StrategyParams(
        fair_value=10000,
        take_width=3,
        clear_width=1,
        disregard_edge=2,  # Increased from 2
        join_edge=4,       # Increased from 3
        default_edge=8,    # Increased from 5
        position_limit=50,
        soft_position_limit=40,
        ema_alpha=0.1    # Slower EMA for stable product
    ),
    Product.KELP: StrategyParams(
        take_width=1,
        clear_width=1,   # Tighter clear width
        adverse_volume=10,
        reversion_beta=-0.229,
        disregard_edge=1,
        join_edge=1,    # Slightly increased
        default_edge=2,  # Increased from 2
        position_limit=50,
        gamma=0.2,        # Adjusted GLFT gamma
        order_amount=25    # Increased order amount
    ),
    Product.SQUID_INK: StrategyParams(
        take_width=1.5,    # Reduced from 2
        clear_width=0.75,  # Reduced from 1
        adverse_volume=3,
        reversion_beta=0.15, # Increased from 0.1
        position_limit=50,
        soft_position_limit=15,
        window_size=5,
        disregard_edge=1,
        join_edge=1.5,    # Increased from 1
        default_edge=3,
        ema_alpha=0.3,     # Faster EMA for volatile product
        gamma=0.4,         # Higher gamma for tighter spreads
        order_amount=30    # Larger order amounts
    )
}

class BaseStrategy:
    def __init__(self, product: str, params: dict):
        self.product = product
        self.position_limit = params.get('position_limit', 50)
        self.soft_position_limit = params.get('soft_position_limit', 40)
        self.take_width = params.get('take_width', 1.5)
        self.clear_width = params.get('clear_width', 0.75)
        self.default_edge = params.get('default_edge', 3)
        self.join_edge = params.get('join_edge', 1.5)
        self.disregard_edge = params.get('disregard_edge', 1)
        self.window_size = params.get('window_size', 10)
        self.position_window = deque(maxlen=self.window_size)
        self.ema_short = None
        self.ema_long = None
        self.price_history = deque(maxlen=20)
        
    def update_ema(self, price: float):
        alpha_short = 0.2  # Fast EMA (5-period equivalent)
        alpha_long = 0.05   # Slow EMA (20-period equivalent)
        
        if self.ema_short is None:
            self.ema_short = price
            self.ema_long = price
        else:
            self.ema_short = alpha_short * price + (1 - alpha_short) * self.ema_short
            self.ema_long = alpha_long * price + (1 - alpha_long) * self.ema_long
    
    def update_position_window(self, position: int):
        self.position_window.append(abs(position) == self.position_limit)
    
    def should_liquidate(self) -> tuple[bool, bool]:
        if len(self.position_window) < self.window_size:
            return False, False
        
        soft_liquidate = (sum(self.position_window) >= self.window_size / 2 
                          and self.position_window[-1])
        hard_liquidate = all(self.position_window)
        return soft_liquidate, hard_liquidate
    
    def get_order_quantities(self, position: int) -> tuple[int, int]:
        to_buy = self.position_limit - position
        to_sell = self.position_limit + position
        return to_buy, to_sell

    def take_orders(self, order_depth: OrderDepth, fair_value: float, position: int) -> tuple[List[Order], int, int]:
        orders = []
        to_buy, to_sell = self.get_order_quantities(position)
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask <= fair_value - self.take_width:
                quantity = min(to_buy, -order_depth.sell_orders[best_ask])
                if quantity > 0:
                    orders.append(Order(self.product, best_ask, quantity))
                    to_buy -= quantity
        
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= fair_value + self.take_width:
                quantity = min(to_sell, order_depth.buy_orders[best_bid])
                if quantity > 0:
                    orders.append(Order(self.product, best_bid, -quantity))
                    to_sell -= quantity
        
        return orders, to_buy, to_sell

    def make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, 
                   to_buy: int, to_sell: int) -> List[Order]:
        orders = []
        pos_ratio = abs(position) / self.position_limit
        
        # Dynamic spread based on position
        spread = max(1, round(self.default_edge * (1 - pos_ratio)))
        
        # Calculate bid/ask prices
        bid = round(fair_value - spread)
        ask = round(fair_value + spread)
        
        # Position-based adjustments
        if position > self.soft_position_limit:
            bid -= round(pos_ratio * 2)
        elif position < -self.soft_position_limit:
            ask += round(pos_ratio * 2)
        
        # Place orders
        if to_buy > 0:
            orders.append(Order(self.product, bid, to_buy))
        if to_sell > 0:
            orders.append(Order(self.product, ask, -to_sell))
        
        return orders

class SquidInkStrategy(BaseStrategy):
    def __init__(self, product: str, params: dict):
        super().__init__(product, params)
        self.momentum_window = 5
        self.volatility_window = 20
        self.price_history = deque(maxlen=self.volatility_window)
        self.momentum = 0
        self.volatility = 0
        
    def calculate_fair_value(self, order_depth: OrderDepth) -> float:
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        self.price_history.append(mid_price)
        
        # Update EMAs
        self.update_ema(mid_price)
        
        # Calculate momentum (5-period)
        if len(self.price_history) >= self.momentum_window:
            self.momentum = mid_price - np.mean(list(self.price_history)[-self.momentum_window:])
        
        # Calculate volatility (20-period std dev)
        if len(self.price_history) >= 2:
            returns = np.diff(self.price_history) / self.price_history[:-1]
            self.volatility = np.std(returns[-self.volatility_window:]) if returns.size > 0 else 0
        
        # Adjust fair value based on momentum and volatility
        momentum_factor = 1 + math.tanh(self.momentum * 10) * 0.005
        volatility_factor = 1 + self.volatility * 2
        fair_value = self.ema_short * momentum_factor * volatility_factor
        
        return fair_value
    
    def make_orders(self, order_depth: OrderDepth, fair_value: float, position: int, 
                   to_buy: int, to_sell: int) -> List[Order]:
        orders = []
        pos_ratio = abs(position) / self.position_limit
        
        # Dynamic parameters based on momentum and volatility
        momentum_strength = abs(self.momentum) / (self.volatility + 1e-6)
        is_strong_trend = momentum_strength > 1.5
        is_uptrend = self.momentum > 0
        
        # Base spread
        spread = max(1, round(self.default_edge * (1 - min(1, momentum_strength/3))))
        
        # Trend-following adjustments
        if is_strong_trend:
            if is_uptrend:
                # More aggressive buying in uptrend
                bid = round(fair_value - spread * 0.5)
                ask = round(fair_value + spread * 2)
                buy_quantity = min(to_buy, 10 + int(momentum_strength * 5))
                sell_quantity = min(to_sell, 5)
            else:
                # More aggressive selling in downtrend
                bid = round(fair_value - spread * 2)
                ask = round(fair_value + spread * 0.5)
                buy_quantity = min(to_buy, 5)
                sell_quantity = min(to_sell, 10 + int(momentum_strength * 5))
        else:
            # Mean-reversion mode
            bid = round(fair_value - spread)
            ask = round(fair_value + spread)
            buy_quantity = min(to_buy, 5)
            sell_quantity = min(to_sell, 5)
        
        # Position management
        if position > self.soft_position_limit:
            bid -= round(pos_ratio * 2)
            ask -= round(pos_ratio * 2)
        elif position < -self.soft_position_limit:
            bid += round(pos_ratio * 2)
            ask += round(pos_ratio * 2)
        
        # Place orders
        if buy_quantity > 0:
            orders.append(Order(self.product, bid, buy_quantity))
        if sell_quantity > 0:
            orders.append(Order(self.product, ask, -sell_quantity))
        
        return orders


class FixedValueStrategy(BaseStrategy):
    def calculate_fair_value(self, state) -> float:
        return self.params.fair_value

class MeanRevertingStrategy(BaseStrategy):
    def __init__(self, product: str, params: StrategyParams):
        super().__init__(product, params)
        self.recent_prices = deque(maxlen=100)
        self.last_price = None
    
    def calculate_fair_value(self, state) -> float:
        order_depth = state.order_depths.get(self.product, None)
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return self.last_price if self.last_price else None
            
        filtered_asks = [p for p, vol in order_depth.sell_orders.items() 
                        if abs(vol) >= self.params.adverse_volume]
        filtered_bids = [p for p, vol in order_depth.buy_orders.items() 
                        if abs(vol) >= self.params.adverse_volume]
        
        mm_ask = min(filtered_asks) if filtered_asks else None
        mm_bid = max(filtered_bids) if filtered_bids else None
        
        if mm_ask and mm_bid:
            current_mid = (mm_ask + mm_bid) / 2
        else:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            current_mid = (best_ask + best_bid) / 2
        
        self.recent_prices.append(current_mid)
        self.last_price = current_mid
        
        # Use EMA for smoother fair value
        ema_value = self.update_ema(current_mid)
        
        beta = self.params.reversion_beta
        if len(self.recent_prices) >= 50:
            beta = self.calculate_dynamic_beta()
        
        if len(self.recent_prices) > 1:
            last_return = (current_mid - self.recent_prices[-2]) / self.recent_prices[-2]
            predicted_return = last_return * beta
            return ema_value * (1 + predicted_return)
        return ema_value
    
    def calculate_dynamic_beta(self) -> float:
        prices = list(self.recent_prices)
        short_window = 10
        long_window = 40
        
        if len(prices) < long_window:
            return self.params.reversion_beta
            
        volatility = np.std(np.diff(prices[-long_window:]))
        vol_factor = 0.01 / (volatility + 1e-6)
        adaptive_beta = self.params.reversion_beta * min(1.5, max(0.5, vol_factor))
        return adaptive_beta

class KelpStrategy(MeanRevertingStrategy):
    def make_orders(self, order_depth, fair_value, position, to_buy, to_sell, soft_liquidate, hard_liquidate):
        # First use GLFT method for KELP
        glft_orders = self.mm_glft(
            order_depth, 
            fair_value, 
            position, 
            to_buy, 
            to_sell,
            gamma=0.1,
            order_amount=20
        )
        
        # Then add standard market making as fallback
        standard_orders = super().make_orders(
            order_depth,
            fair_value,
            position,
            to_buy - sum(o.quantity for o in glft_orders if o.quantity > 0),
            to_sell - sum(-o.quantity for o in glft_orders if o.quantity < 0),
            soft_liquidate,
            hard_liquidate
        )
        
        return glft_orders + standard_orders
    



class Trader:
    def __init__(self):
        self.strategies = {
            Product.RAINFOREST_RESIN: FixedValueStrategy(Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN]),
            Product.KELP: KelpStrategy(Product.KELP, PARAMS[Product.KELP]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, PARAMS[Product.SQUID_INK])
        }
        self.trader_data = {}
    
    def run(self, state: TradingState):
        if state.traderData:
            self.trader_data = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0
        
        for product, strategy in self.strategies.items():
            if product not in state.order_depths:
                continue
                
            position = state.position.get(product, 0)
            strategy.update_position_window(position)
            soft_liquidate, hard_liquidate = strategy.should_liquidate()
            
            fair_value = strategy.calculate_fair_value(state)
            if fair_value is None:
                continue
            
            # Execute trading logic in phases
            take_orders, to_buy, to_sell = strategy.take_orders(
                state.order_depths[product],
                fair_value,
                position
            )
            
            clear_orders, to_buy, to_sell = strategy.clear_orders(
                state.order_depths[product],
                fair_value,
                position,
                to_buy,
                to_sell
            )
            
            make_orders = strategy.make_orders(
                state.order_depths[product],
                fair_value,
                position,
                to_buy,
                to_sell,
                soft_liquidate,
                hard_liquidate
            )
            
            result[product] = take_orders + clear_orders + make_orders
        
        trader_data = jsonpickle.encode(self.trader_data)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data