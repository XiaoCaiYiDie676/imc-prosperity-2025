import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import jsonpickle
import json
from math import log, sqrt
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datamodel import Order, OrderDepth, TradingState, ProsperityEncoder, Symbol, ConversionObservation, Listing, Trade, Observation


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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()



class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    JAMS = "JAMS"
    CROISSANTS = "CROISSANTS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

@dataclass
class StrategyParams:
    take_width: float = 1.0
    clear_width: float = 1.0
    adverse_volume: int = 0
    reversion_beta: float = 0.2
    disregard_edge: float = 1.0
    join_edge: float = 2.0
    default_edge: float = 4.0
    position_limit: int = 50
    soft_position_limit: int = 40
    window_size: int = 10
    fair_value: float = None
    ema_alpha: float = 0.2
    gamma: float = 0.1
    order_amount: int = 20

PARAMS = {
    Product.RAINFOREST_RESIN: StrategyParams(fair_value=10000, take_width=3, clear_width=0, disregard_edge=2, join_edge=4, default_edge=8, position_limit=50, soft_position_limit=40, ema_alpha=0.1),
    Product.KELP: StrategyParams(take_width=1, clear_width=1, adverse_volume=10, reversion_beta=-0.229, disregard_edge=1, join_edge=1, default_edge=2, position_limit=50, gamma=0.3, order_amount=25),
    Product.SQUID_INK: StrategyParams(take_width=1.5, clear_width=0.75, adverse_volume=3, reversion_beta=0.3, position_limit=50, soft_position_limit=15, window_size=5, disregard_edge=1, join_edge=1.5, default_edge=3, ema_alpha=0.3, gamma=0.4, order_amount=50),
    Product.JAMS: StrategyParams(take_width=1, clear_width=1, adverse_volume=10, reversion_beta=-0.229, disregard_edge=1, join_edge=1, default_edge=2, position_limit=350, gamma=0.3, order_amount=25),
    Product.CROISSANTS: StrategyParams(take_width=1, clear_width=1, position_limit=250, disregard_edge=1, join_edge=1, default_edge=2),
    Product.DJEMBES: StrategyParams(take_width=1, clear_width=1, position_limit=60, disregard_edge=1, join_edge=1, default_edge=2),
    Product.PICNIC_BASKET1: StrategyParams(position_limit=60),
    Product.PICNIC_BASKET2: StrategyParams(position_limit=100),
    Product.VOLCANIC_ROCK: StrategyParams(position_limit=400),
    Product.VOLCANIC_ROCK_VOUCHER_9500: StrategyParams(position_limit=200),
    Product.VOLCANIC_ROCK_VOUCHER_9750: StrategyParams(position_limit=200),
    Product.VOLCANIC_ROCK_VOUCHER_10000: StrategyParams(position_limit=200),
    Product.VOLCANIC_ROCK_VOUCHER_10250: StrategyParams(position_limit=200),
    Product.VOLCANIC_ROCK_VOUCHER_10500: StrategyParams(position_limit=100),
    Product.MAGNIFICENT_MACARONS: StrategyParams(position_limit=75),
}

class PriceAnalyzer:
    def __init__(self):
        self.fair_values: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = defaultdict(list)
        self.ema_factors: Dict[str, float] = {}
        self.spread_history: Dict[str, List[float]] = defaultdict(list)
        self._last_mid_vwaps: Dict[str, float] = {}

    def calculate_vwap(self, orders: Dict[float, int], is_buy_side: bool) -> float:
        if not orders:
            return 0
        price_levels = sorted(orders.keys(), reverse=is_buy_side)[:3]
        volumes = np.array([orders[p] for p in price_levels])
        prices = np.array(price_levels)
        total_volume = volumes.sum()
        return (prices * volumes).sum() / total_volume if total_volume else 0

    def calculate_dynamic_alpha(self, product: str) -> float:
        prices = self.price_history[product][-5:]
        if len(prices) < 5:
            return 0.2
        prices = np.array(prices)
        volatility = prices.max() - prices.min()
        avg_price = prices.mean()
        normalized_volatility = volatility / avg_price if avg_price else 0
        return max(0.15, min(0.4, 0.3 / (1 + normalized_volatility * 8)))

    def is_outlier(self, product: str, price: float) -> bool:
        prices = self.price_history[product][-10:]
        if len(prices) < 10:
            return False
        prices = np.array(prices)
        mean_price = prices.mean()
        std_dev = prices.std(ddof=1) if len(prices) > 1 else 0
        return abs(price - mean_price) > 2.5 * std_dev

    def update_fair_values(self, order_depths: Dict[str, OrderDepth]) -> None:
        for product, order_depth in order_depths.items():
            if product in self._last_mid_vwaps and not order_depth.buy_orders and not order_depth.sell_orders:
                continue
            bid_vwap = self.calculate_vwap(order_depth.buy_orders, True)
            ask_vwap = self.calculate_vwap(order_depth.sell_orders, False)
            mid_vwap = (bid_vwap + ask_vwap) / 2 if bid_vwap and ask_vwap else bid_vwap or ask_vwap
            if not mid_vwap:
                continue
            self._last_mid_vwaps[product] = mid_vwap
            self.price_history[product].append(mid_vwap)
            if len(self.price_history[product]) > 10:
                self.price_history[product].pop(0)
            if self.is_outlier(product, mid_vwap):
                mid_vwap = self.ema_factors.get(product, mid_vwap)
            alpha = self.calculate_dynamic_alpha(product)
            long_ema = self.ema_factors.get(product + '_long', mid_vwap)
            long_ema = 0.05 * mid_vwap + (1 - 0.05) * long_ema
            self.ema_factors[product + '_long'] = long_ema
            self.ema_factors[product] = alpha * mid_vwap + (1 - alpha) * self.ema_factors.get(product, mid_vwap)
            self.fair_values[product] = (self.ema_factors[product] + long_ema) / 2

    def calculate_basket_values(self, components: Dict[str, Dict[str, int]]) -> None:
        for basket, composition in components.items():
            try:
                basket_value = sum(qty * self.fair_values[component]
                                   for component, qty in composition.items()
                                   if component in self.fair_values)
                self.fair_values[f"DERIVED_{basket}"] = basket_value
            except KeyError:
                continue

    def update_spread_history(self, baskets: List[str]) -> None:
        for basket in baskets:
            derived_key = f"DERIVED_{basket}"
            if derived_key in self.fair_values and basket in self.fair_values:
                spread = self.fair_values[basket] - self.fair_values[derived_key]
                self.spread_history[f"SPREAD_{basket}"].append(spread)
                if len(self.spread_history[f"SPREAD_{basket}"]) > 10:
                    self.spread_history[f"SPREAD_{basket}"].pop(0)

class BaseStrategy(ABC):
    def __init__(self, product: str, params: StrategyParams):
        self.product = product
        self.params = params
        self.position_window = []
        self.ema = None
        self.price_history = defaultdict(list)

    def update_ema(self, new_value: float) -> float:
        self.ema = (self.params.ema_alpha * new_value + (1 - self.params.ema_alpha) * self.ema
                    if self.ema is not None else new_value)
        return self.ema

    @abstractmethod
    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        pass

    def update_position_window(self, position: int):
        self.position_window.append(abs(position) >= self.params.position_limit)
        if len(self.position_window) > self.params.window_size:
            self.position_window.pop(0)

    def should_liquidate(self) -> tuple[bool, bool]:
        if len(self.position_window) < self.params.window_size:
            return False, False
        soft_liquidate = sum(self.position_window) >= self.params.window_size / 2 and self.position_window[-1]
        hard_liquidate = all(self.position_window)
        return soft_liquidate, hard_liquidate

    def get_order_quantities(self, position: int) -> tuple[int, int]:
        to_buy = self.params.position_limit - position
        to_sell = self.params.position_limit + position
        return to_buy, to_sell

    def check_stop_loss(self, order_depth: OrderDepth, fair_value: float, position: int) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        if position > 0 and best_bid < fair_value * 0.95:
            qty = min(position, order_depth.buy_orders[best_bid])
            orders.append(Order(self.product, best_bid, -qty))
        elif position < 0 and best_ask > fair_value * 1.05:
            qty = min(-position, -order_depth.sell_orders[best_ask])
            orders.append(Order(self.product, best_ask, qty))
        return orders

    def take_orders(self, order_depth: OrderDepth, fair_value: float, position: int) -> tuple[List[Order], int, int]:
        orders = []
        to_buy, to_sell = self.get_order_quantities(position)
        volatility = np.std(self.price_history.get(self.product, [-1])[-10:]) if len(self.price_history.get(self.product, [])) >= 10 else 1
        dynamic_take_width = max(0.5, self.params.take_width * (1 + volatility / 2))
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_vol = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - dynamic_take_width:
                quantity = min(to_buy, best_ask_vol)
                if quantity > 0:
                    orders.append(Order(self.product, best_ask, quantity))
                    to_buy -= quantity
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + dynamic_take_width:
                quantity = min(to_sell, best_bid_vol)
                if quantity > 0:
                    orders.append(Order(self.product, best_bid, -quantity))
                    to_sell -= quantity
        return orders, to_buy, to_sell

    def clear_orders(self, order_depth: OrderDepth, fair_value: float, position: int,
                    to_buy: int, to_sell: int) -> tuple[List[Order], int, int]:
        orders = []
        position_after_take = position + (self.params.position_limit - to_buy) - (self.params.position_limit - to_sell)
        if position_after_take > 0:
            clear_price = round(fair_value + self.params.clear_width)
            clear_vol = sum(vol for price, vol in order_depth.buy_orders.items() if price >= clear_price)
            quantity = min(position_after_take, clear_vol, to_sell)
            if quantity > 0:
                orders.append(Order(self.product, clear_price, -quantity))
                to_sell -= quantity
        if position_after_take < 0:
            clear_price = round(fair_value - self.params.clear_width)
            clear_vol = sum(-vol for price, vol in order_depth.sell_orders.items() if price <= clear_price)
            quantity = min(-position_after_take, clear_vol, to_buy)
            if quantity > 0:
                orders.append(Order(self.product, clear_price, quantity))
                to_buy -= quantity
        return orders, to_buy, to_sell

    def mm_glft(self, order_depth: OrderDepth, fair_value: float, position: int,
                to_buy: int, to_sell: int) -> List[Order]:
        if self.product not in [Product.KELP, Product.SQUID_INK]:
            return []
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
        gamma = self.params.gamma
        order_amount = self.params.order_amount
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        q = position / order_amount
        kappa_b = 1 / max((fair_value - best_bid) - 1, 1)
        kappa_a = 1 / max((best_ask - fair_value) - 1, 1)
        A_b = A_a = 0.25
        delta_b = (1/gamma * log(1 + gamma/kappa_b) +
                   (-0/(gamma * 0.3959**2) + (2*q + 1)/2) *
                   sqrt((0.3959**2 * gamma)/(2 * kappa_b * A_b) *
                        (1 + gamma/kappa_b)**(1 + kappa_b/gamma)))
        delta_a = (1/gamma * log(1 + gamma/kappa_a) +
                   (0/(gamma * 0.3959**2) - (2*q - 1)/2) *
                   sqrt((0.3959**2 * gamma)/(2 * kappa_a * A_a) *
                        (1 + gamma/kappa_a)**(1 + kappa_a/gamma)))
        p_b = round(fair_value - delta_b)
        p_a = round(fair_value + delta_a)
        p_b = min(p_b, fair_value, best_bid + 1)
        p_a = max(p_a, fair_value, best_ask - 1)
        buy_amount = min(order_amount, to_buy)
        sell_amount = min(order_amount, to_sell)
        if buy_amount > 0:
            orders.append(Order(self.product, int(p_b), int(buy_amount)))
        if sell_amount > 0:
            orders.append(Order(self.product, int(p_a), -int(sell_amount)))
        return orders

    def make_orders(self, order_depth: OrderDepth, fair_value: float, position: int,
                   to_buy: int, to_sell: int, soft_liquidate: bool, hard_liquidate: bool) -> List[Order]:
        orders = []
        asks_above = [p for p in order_depth.sell_orders.keys() if p > fair_value + self.params.disregard_edge]
        bids_below = [p for p in order_depth.buy_orders.keys() if p < fair_value - self.params.disregard_edge]
        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None
        base_ask = round(fair_value + self.params.default_edge)
        ask = best_ask_above if best_ask_above and abs(best_ask_above - fair_value) <= self.params.join_edge else \
              best_ask_above - 1 if best_ask_above else base_ask
        base_bid = round(fair_value - self.params.default_edge)
        bid = best_bid_below if best_bid_below and abs(fair_value - best_bid_below) <= self.params.join_edge else \
              best_bid_below + 1 if best_bid_below else base_bid
        if hard_liquidate:
            ask = bid = round(fair_value)
        elif soft_liquidate:
            ask = round(fair_value + 4)
            bid = round(fair_value - 4)
        pos_ratio = abs(position) / self.params.position_limit
        if position > self.params.soft_position_limit:
            bid += round(pos_ratio * 2)
        elif position < -self.params.soft_position_limit:
            ask -= round(pos_ratio * 2)
        volume_scale = min(1.0, 0.5 + pos_ratio)
        if to_buy > 0:
            orders.append(Order(self.product, bid, int(to_buy * volume_scale)))
        if to_sell > 0:
            orders.append(Order(self.product, ask, -int(to_sell * volume_scale)))
        return orders
    
class TraderState:
    price_history: List[float] = None
    last_trade_price: float = 0.0
    trend: float = 0.0
    trend_strength: float = 0.0
    volatility: float = 5.0
    pnl: float = 0.0
    position: int = 0
    daily_conversions: int = 0
    current_day: int = 0
    debug_info: Dict[str, any] = None

    def __post_init__(self):
        self.price_history = self.price_history or []
        self.debug_info = self.debug_info or {}
    
class MacaronTrader:
    def __init__(self):
        self.symbol = "MAGNIFICENT_MACARONS"
        self.position_limit = 75
        self.conversion_limit = 10
        self.state = TraderState()
        
    def calculate_fair_value(self, conv_obs: ConversionObservation) -> float:
        """Calculate fair value based on conversion observations"""
        if not conv_obs:
            return 0
        
        # Base fair value is midpoint between bid and ask, adjusted for costs
        midpoint = (conv_obs.bidPrice + conv_obs.askPrice) / 2
        transport_cost = conv_obs.transportFees
        tariff_cost = (conv_obs.importTariff + conv_obs.exportTariff) / 2
        
        # Adjust for sugar price (normalized to 100 as baseline)
        sugar_factor = conv_obs.sugarPrice / 100
        
        # Adjust for sunlight (normalized to 10 as baseline)
        sunlight_factor = conv_obs.sunlightIndex / 10
        
        # Calculate final fair value
        fair_value = midpoint - transport_cost - tariff_cost
        fair_value *= (1 + 0.05 * (sugar_factor - 1) + 0.03 * (sunlight_factor - 1))
        
        return fair_value

    def generate_orders(self, state: TradingState) -> List[Order]:
        """Generate simple arbitrage orders between spot and conversion prices"""
        orders = []
        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths.get(self.symbol, OrderDepth())
        conv_obs = state.observations.conversionObservations.get(self.symbol)
        
        if not order_depth.buy_orders or not order_depth.sell_orders or not conv_obs:
            return orders
            
        fair_value = self.calculate_fair_value(conv_obs)
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate effective conversion prices including costs
        effective_conv_buy = conv_obs.askPrice + conv_obs.importTariff + conv_obs.transportFees
        effective_conv_sell = conv_obs.bidPrice - conv_obs.exportTariff - conv_obs.transportFees
        
        # Arbitrage opportunities
        # Sell to market when better than conversion
        if best_bid > effective_conv_sell:
            max_sell = min(
                order_depth.buy_orders[best_bid],
                self.position_limit - position
            )
            if max_sell > 0:
                orders.append(Order(self.symbol, best_bid, -max_sell))
                
        # Buy from market when better than conversion
        if best_ask < effective_conv_buy:
            max_buy = min(
                -order_depth.sell_orders[best_ask],
                self.position_limit + position
            )
            if max_buy > 0:
                orders.append(Order(self.symbol, best_ask, max_buy))
                
        return orders

    def determine_conversions(self, state: TradingState) -> int:
        """Determine how many units to convert at the end"""
        position = state.position.get(self.symbol, 0)
        if position == 0:
            return 0
            
        # Force conversion at the end (up to conversion_limit)
        if state.timestamp >= 999000:  # Adjust based on competition end time
            return -min(abs(position), self.conversion_limit)
            
        # Early partial conversion if position is large
        if abs(position) > 50 and state.timestamp >= 950000:  # Last 10% of session
            return -int(np.sign(position) * min(self.conversion_limit, abs(position) // 5))
            
        return 0

class OptionVoucherStrategy(BaseStrategy):
    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        return price_analyzer.fair_values.get(self.product, 0)

class FixedValueStrategy(BaseStrategy):
    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        return self.params.fair_value

class MeanRevertingStrategy(BaseStrategy):
    def __init__(self, product: str, params: StrategyParams):
        super().__init__(product, params)
        self.recent_prices = []
        self.last_price = None

    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        order_depth = state.order_depths.get(self.product)
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return self.last_price
        filtered_asks = [p for p, vol in order_depth.sell_orders.items() if abs(vol) >= self.params.adverse_volume]
        filtered_bids = [p for p, vol in order_depth.buy_orders.items() if abs(vol) >= self.params.adverse_volume]
        mm_ask = min(filtered_asks) if filtered_asks else min(order_depth.sell_orders.keys())
        mm_bid = max(filtered_bids) if filtered_bids else max(order_depth.buy_orders.keys())
        current_mid = (mm_ask + mm_bid) / 2
        self.recent_prices.append(current_mid)
        if len(self.recent_prices) > 100:
            self.recent_prices.pop(0)
        self.last_price = current_mid
        self.price_history[self.product].append(current_mid)
        ema_value = self.update_ema(current_mid)
        beta = self.calculate_dynamic_beta() if len(self.recent_prices) >= 50 else self.params.reversion_beta
        if len(self.recent_prices) > 1:
            last_return = (current_mid - self.recent_prices[-2]) / self.recent_prices[-2]
            predicted_return = last_return * beta
            return ema_value * (1 + predicted_return)
        return ema_value

    def calculate_dynamic_beta(self) -> float:
        if len(self.recent_prices) < 40:
            return self.params.reversion_beta
        prices = np.array(self.recent_prices[-40:])
        price_diffs = np.diff(prices)
        volatility = price_diffs.std(ddof=1) if len(price_diffs) > 1 else 1
        vol_factor = 0.01 / (volatility + 1e-6)
        return self.params.reversion_beta * min(1.5, max(0.5, vol_factor))

class KelpStrategy(MeanRevertingStrategy):
    def make_orders(self, order_depth, fair_value, position, to_buy, to_sell, soft_liquidate, hard_liquidate):
        glft_orders = self.mm_glft(order_depth, fair_value, position, to_buy, to_sell)
        standard_orders = super().make_orders(
            order_depth, fair_value, position,
            to_buy - sum(o.quantity for o in glft_orders if o.quantity > 0),
            to_sell - sum(-o.quantity for o in glft_orders if o.quantity < 0),
            soft_liquidate, hard_liquidate
        )
        return glft_orders + standard_orders

class SquidInkStrategy(MeanRevertingStrategy):
    def __init__(self, product: str, params: StrategyParams):
        super().__init__(product, params)
        self.short_window = 5
        self.short_prices = []
        self.short_returns = []
        self.volatility_window = []

    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        order_depth = state.order_depths.get(self.product)
        if not order_depth or not order_depth.sell_orders or not order_depth.buy_orders:
            return self.last_price
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) / 2
        self.short_prices.append(mid_price)
        if len(self.short_prices) > self.short_window:
            self.short_prices.pop(0)
        self.last_price = mid_price
        self.price_history[self.product].append(mid_price)
        if len(self.short_prices) < 2:
            return mid_price
        prices = np.array(self.short_prices)
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        self.short_returns.extend(returns)
        if len(self.short_returns) > self.short_window:
            self.short_returns = self.short_returns[-self.short_window:]
        volatility = returns.std(ddof=1) if len(returns) > 1 else 0
        self.volatility_window.append(volatility)
        if len(self.volatility_window) > 10:
            self.volatility_window.pop(0)
        ema_value = self.update_ema(mid_price)
        if len(self.volatility_window) > 0:
            avg_volatility = np.mean(self.volatility_window)
            self.params.gamma = max(0.2, min(0.6, 0.4 + avg_volatility * 2))
        if len(self.short_returns) > 0:
            last_return = self.short_returns[-1]
            predicted_return = last_return * self.params.reversion_beta
            volatility_factor = 1 + avg_volatility if len(self.volatility_window) > 0 else 1
            return ema_value * (1 + predicted_return * volatility_factor)
        return ema_value

    def make_orders(self, order_depth: OrderDepth, fair_value: float, position: int,
                   to_buy: int, to_sell: int, soft_liquidate: bool, hard_liquidate: bool) -> List[Order]:
        glft_orders = self.mm_glft(order_depth, fair_value, position, to_buy, to_sell)
        remaining_to_buy = to_buy - sum(o.quantity for o in glft_orders if o.quantity > 0)
        remaining_to_sell = to_sell - sum(-o.quantity for o in glft_orders if o.quantity < 0)
        pos_ratio = abs(position) / self.params.position_limit
        if pos_ratio > 0.7:
            self.params.clear_width *= 1.5
        elif pos_ratio < 0.3:
            self.params.clear_width = max(0.5, self.params.clear_width * 0.9)
        standard_orders = super().make_orders(
            order_depth, fair_value, position, remaining_to_buy, remaining_to_sell, soft_liquidate, hard_liquidate
        )
        return glft_orders + standard_orders

class ComponentStrategy(BaseStrategy):
    def calculate_fair_value(self, state: TradingState, price_analyzer: PriceAnalyzer) -> float:
        return price_analyzer.fair_values.get(self.product, 0)

    def generate_orders(self, state: TradingState, order_depth: OrderDepth, price_analyzer: PriceAnalyzer) -> List[Order]:
        if self.product not in price_analyzer.fair_values:
            return []
        current_pos = state.position.get(self.product, 0)
        fair_value = price_analyzer.fair_values[self.product]
        orders = []
        max_long = int((self.params.position_limit - current_pos) * 0.8)
        max_short = int((self.params.position_limit + current_pos) * 0.8)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            ask_volume = abs(order_depth.sell_orders[best_ask])
            profit_margin = fair_value - best_ask
            if profit_margin > 3:
                buy_qty = min(ask_volume, max_long, 40 if profit_margin <= 6 else max_long * 2)
                if buy_qty > 0:
                    orders.append(Order(self.product, best_ask, buy_qty))
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            bid_volume = order_depth.buy_orders[best_bid]
            profit_margin = best_bid - fair_value
            if profit_margin > 3:
                sell_qty = min(bid_volume, max_short, 40 if profit_margin <= 6 else max_short * 2)
                if sell_qty > 0:
                    orders.append(Order(self.product, best_bid, -sell_qty))
        if not orders:
            spread = 1
            if max_long > 0:
                orders.append(Order(self.product, int(fair_value - spread), min(40, max_long)))
            if max_short > 0:
                orders.append(Order(self.product, int(fair_value + spread), -min(40, max_short)))
        return orders

class BasketArbitrageStrategy:
    def __init__(self, price_analyzer: PriceAnalyzer):
        self.price_analyzer = price_analyzer
        self.min_profit_threshold = 10
        self.components = {
            Product.PICNIC_BASKET1: {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1},
            Product.PICNIC_BASKET2: {Product.CROISSANTS: 4, Product.JAMS: 2}
        }

    def generate_orders(self, state: TradingState, product: str) -> List[Order]:
        orders = []
        derived_key = f"DERIVED_{product}"
        if derived_key not in self.price_analyzer.fair_values:
            return orders
        order_depth = state.order_depths.get(product, OrderDepth())
        current_pos = state.position.get(product, 0)
        derived_value = self.price_analyzer.fair_values[derived_key]
        volatility = np.std(self.price_analyzer.price_history[product][-10:]) if len(self.price_analyzer.price_history[product]) >= 10 else 1
        dynamic_threshold = max(5, min(15, 8 + volatility * 0.5))
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < derived_value - dynamic_threshold:
                max_buy = PARAMS[product].position_limit - current_pos
                buy_qty = min(abs(order_depth.sell_orders[best_ask]), max_buy)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    orders.extend(self._generate_hedge_orders(state, product, -1, buy_qty))
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid > derived_value + dynamic_threshold:
                max_sell = PARAMS[product].position_limit + current_pos
                sell_qty = min(order_depth.buy_orders[best_bid], max_sell)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    orders.extend(self._generate_hedge_orders(state, product, 1, sell_qty))
        return orders

    def _generate_hedge_orders(self, state: TradingState, basket: str, hedge_sign: int, qty: int) -> List[Order]:
        hedge_orders = defaultdict(list)
        for component, ratio in self.components[basket].items():
            if ratio == 0 or component not in state.order_depths:
                continue
            order_depth = state.order_depths[component]
            current_pos = state.position.get(component, 0)
            max_qty = PARAMS[component].position_limit - current_pos if hedge_sign > 0 else PARAMS[component].position_limit + current_pos
            hedge_qty = min(qty * ratio, max_qty)
            if hedge_sign > 0 and order_depth.sell_orders:
                best_ask = min(order_depth.sell_orders.keys())
                qty = min(abs(order_depth.sell_orders[best_ask]), hedge_qty)
                if qty > 0:
                    hedge_orders[component].append(Order(component, best_ask, qty))
            elif hedge_sign < 0 and order_depth.buy_orders:
                best_bid = max(order_depth.buy_orders.keys())
                qty = min(order_depth.buy_orders[best_bid], hedge_qty)
                if qty > 0:
                    hedge_orders[component].append(Order(component, best_bid, -qty))
        return [order for orders in hedge_orders.values() for order in orders]

class OptionsStrategy:
    def __init__(self):
        self.strike_prices = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        self.transaction_cost = 0.001
        self.tariff = 0.5
        self.trader_data = {'trade_count': 0, 'feature_history': [], 'last_mid_prices': {}, 'position_cost': {}, 'iv_history': []}

    def calculate_net_profit(self, buy_price, sell_price, size):
        gross_profit = (sell_price - buy_price) * size
        return gross_profit - (self.transaction_cost * (buy_price + sell_price) + self.tariff) * size

    def generate_orders(self, state: TradingState, price_analyzer: PriceAnalyzer) -> tuple[Dict[str, List[Order]], int]:
        result = defaultdict(list)
        conversions = 0
        estimated_profit = 0
        trader_data = self.trader_data
        trade_count = trader_data.get('trade_count', 0)
        feature_history = trader_data.get('feature_history', [])[-10:]
        last_mid_prices = trader_data.get('last_mid_prices', {})
        position_cost = trader_data.get('position_cost', {})
        iv_history = trader_data.get('iv_history', [])[-5:]

        try:
            current_data = {}
            products = [Product.VOLCANIC_ROCK] + list(self.strike_prices.keys())
            for product in products:
                order_depth = state.order_depths.get(product, OrderDepth())
                buy_orders = order_depth.buy_orders
                sell_orders = order_depth.sell_orders
                best_bid = max(buy_orders.keys()) if buy_orders else 0
                best_ask = min(sell_orders.keys()) if sell_orders else 0
                mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else last_mid_prices.get(product, 1)
                if mid_price:
                    last_mid_prices[product] = mid_price
                current_data[product] = {
                    'mid_price': mid_price, 'best_bid': best_bid, 'best_ask': best_ask,
                    'bid_volume': buy_orders.get(best_bid, 0), 'ask_volume': sell_orders.get(best_ask, 0),
                    'depth': order_depth
                }

            vr_mid = current_data.get(Product.VOLCANIC_ROCK, {}).get('mid_price', 0)
            if not vr_mid:
                return result, conversions

            p_9500 = Product.VOLCANIC_ROCK_VOUCHER_9500
            p_9750 = Product.VOLCANIC_ROCK_VOUCHER_9750
            p_10000 = Product.VOLCANIC_ROCK_VOUCHER_10000
            p_10500 = Product.VOLCANIC_ROCK_VOUCHER_10500
            feature = {
                'vr_mid_price': vr_mid,
                'spread_9500_9750': current_data[p_9500]['best_bid'] - current_data[p_9750]['best_ask'] if p_9500 in current_data else 0
            }
            feature_history.append(feature)
            if len(feature_history) > 10:
                feature_history.pop(0)

            positions = state.position.copy()
            for p in current_data:
                positions[p] = positions.get(p, 0)
                position_cost[p] = position_cost.get(p, current_data[p]['mid_price'])

            vr_history = np.array([f['vr_mid_price'] for f in feature_history])
            sma_20 = vr_history.mean() if len(vr_history) > 0 else vr_mid
            is_downtrend = vr_mid < sma_20 * 0.99
            volatility = vr_history.std(ddof=1) if len(vr_history) > 1 else 1

            tte = (60000 - state.timestamp) / 60000 if state.timestamp < 60000 else 0.01

            iv_data = {}
            for voucher in self.strike_prices:
                if voucher in current_data and current_data[voucher]['mid_price'] > 0:
                    fair_value = price_analyzer.fair_values.get(voucher, current_data[voucher]['mid_price'])
                    order_depth = current_data[voucher]['depth']
                    position = positions.get(voucher, 0)
                    strategy = state.strategies.get(voucher, ComponentStrategy(voucher, PARAMS[voucher]))
                    stop_loss_orders = strategy.check_stop_loss(order_depth, fair_value, position)
                    result[voucher].extend(stop_loss_orders)
                    K = self.strike_prices[voucher]
                    St = vr_mid
                    Vt = current_data[voucher]['mid_price']
                    m_t = log(K / St) / sqrt(tte) if tte > 0 and St > 0 else 0
                    v_t = (Vt / St) * sqrt(2 * np.pi / tte) * 0.9 if tte > 0 and St > 0 else 0
                    iv_data[voucher] = {'m_t': m_t, 'v_t': v_t}

            iv_values = [iv_data[v]['v_t'] for v in iv_data]
            base_iv = np.mean(iv_values) if iv_values else 0
            iv_history.append(base_iv)
            if len(iv_history) > 5:
                iv_history.pop(0)
            iv_increasing = np.mean(iv_values[-2:]) - np.mean(iv_values[:2]) > 0 if len(iv_history) >= 2 else False

            if volatility > 3:
                logger.print("High volatility, reducing option arbitrage")
            else:
                if p_9750 in positions and positions[p_9750] > 0:
                    depth_9750 = current_data[p_9750]['depth']
                    best_bid_9750 = current_data[p_9750]['best_bid']
                    size = min(positions[p_9750], abs(depth_9750.buy_orders.get(best_bid_9750, 0)), 5 if volatility > 3 else 10)
                    if size > 0:
                        result[p_9750].append(Order(p_9750, best_bid_9750, -size))
                        positions[p_9750] -= size
                        trade_count += 1

                if p_9500 in current_data and p_9750 in current_data:
                    depth_9750 = current_data[p_9750]['depth']
                    depth_9500 = current_data[p_9500]['depth']
                    best_bid_9500 = current_data[p_9500]['best_bid']
                    best_ask_9750 = current_data[p_9750]['best_ask']
                    spread = best_bid_9500 - best_ask_9750
                    arb_threshold = max(2, 3.5 + self.transaction_cost * (best_bid_9500 + best_ask_9750) + volatility * 0.03)
                    if spread > arb_threshold and not is_downtrend:
                        size = min(5 if volatility > 3 else 10, abs(depth_9750.sell_orders.get(best_ask_9750, 0)),
                                   abs(depth_9500.buy_orders.get(best_bid_9500, 0)),
                                   PARAMS[p_9750].position_limit - positions[p_9750],
                                   PARAMS[p_9500].position_limit + positions[p_9500])
                        if size > 0:
                            result[p_9750].append(Order(p_9750, best_ask_9750, size))
                            result[p_9500].append(Order(p_9500, best_bid_9500, -size))
                            positions[p_9750] += size
                            positions[p_9500] -= size
                            estimated_profit += self.calculate_net_profit(best_ask_9750, best_bid_9500, size)
                            trade_count += 1

                if p_9750 in iv_data and p_10500 in iv_data:
                    v_t_9750 = iv_data[p_9750]['v_t']
                    v_t_10500 = iv_data[p_10500]['v_t']
                    if v_t_10500 > v_t_9750 * 1.1:
                        depth_9750 = current_data[p_9750]['depth']
                        depth_10500 = current_data[p_10500]['depth']
                        best_bid_9750 = current_data[p_9750]['best_bid']
                        best_ask_10500 = current_data[p_10500]['best_ask']
                        size = min(5 if volatility > 3 else 10, abs(depth_10500.sell_orders.get(best_ask_10500, 0)),
                                   abs(depth_9750.buy_orders.get(best_bid_9750, 0)),
                                   PARAMS[p_10500].position_limit + positions[p_10500],
                                   PARAMS[p_9750].position_limit - positions[p_9750])
                        if size > 0:
                            result[p_10500].append(Order(p_10500, best_ask_10500, -size))
                            result[p_9750].append(Order(p_9750, best_bid_9750, size))
                            positions[p_10500] -= size
                            positions[p_9750] += size
                            estimated_profit += self.calculate_net_profit(best_ask_10500, best_bid_9750, size)
                            trade_count += 1

                if p_10000 in current_data and abs(iv_data[p_10000]['v_t'] - np.mean(iv_history[-3:])) > 0.1:
                    depth_10000 = current_data[p_10000]['depth']
                    best_bid_10000 = current_data[p_10000]['best_bid']
                    best_ask_10000 = current_data[p_10000]['best_ask']
                    if iv_increasing and positions[p_10000] < 100:
                        size = min(5, abs(depth_10000.sell_orders.get(best_ask_10000, 0)),
                                   PARAMS[p_10000].position_limit - positions[p_10000])
                        if size > 0:
                            result[p_10000].append(Order(p_10000, best_ask_10000, size))
                            positions[p_10000] += size
                            trade_count += 1
                    elif not iv_increasing and positions[p_10000] > -100:
                        size = min(5, abs(depth_10000.buy_orders.get(best_bid_10000, 0)),
                                   PARAMS[p_10000].position_limit + positions[p_10000])
                        if size > 0:
                            result[p_10000].append(Order(p_10000, best_bid_10000, -size))
                            positions[p_10000] -= size
                            trade_count += 1

            if p_10000 in positions and abs(positions[p_10000]) > 0:
                hedge_ratio = 0.5 if iv_increasing else 1.0
                hedge_size = -min(abs(positions[p_10000]), PARAMS[Product.VOLCANIC_ROCK].position_limit - abs(positions.get(Product.VOLCANIC_ROCK, 0)))
                if hedge_size != 0 and vr_mid < sma_20:
                    price = current_data[Product.VOLCANIC_ROCK]['best_bid'] if hedge_size < 0 else current_data[Product.VOLCANIC_ROCK]['best_ask']
                    result[Product.VOLCANIC_ROCK].append(Order(Product.VOLCANIC_ROCK, price, hedge_size))
                    positions[Product.VOLCANIC_ROCK] = positions.get(Product.VOLCANIC_ROCK, 0) + hedge_size
                    estimated_profit -= self.tariff * abs(hedge_size)

            for voucher in self.strike_prices:
                if voucher in state.observations.conversionObservations and voucher in positions:
                    conv = state.observations.conversionObservations[voucher]
                    expected_profit = conv.bidPrice - conv.askPrice - (conv.importTariff + conv.transportFees)
                    if expected_profit > 0 and positions[voucher] < -5:
                        conv_qty = min(abs(positions[voucher]), 5)
                        conversions += conv_qty
                        estimated_profit -= (conv.importTariff + conv.transportFees) * conv_qty
                        logger.print(f"Conversion for {voucher}: qty={conv_qty}, cost={conv.importTariff + conv.transportFees}")

            self.trader_data.update({
                'trade_count': trade_count,
                'feature_history': feature_history,
                'last_mid_prices': last_mid_prices,
                'position_cost': position_cost,
                'iv_history': iv_history
            })
        except Exception as e:
            logger.print(f"Error in OptionsStrategy.generate_orders: {e}")
        return result, conversions

class Trader:
    def __init__(self):
        self.price_analyzer = PriceAnalyzer()
        self.basket_arbitrage = BasketArbitrageStrategy(self.price_analyzer)
        self.options_strategy = OptionsStrategy()
        self.strategies = {
            Product.RAINFOREST_RESIN: FixedValueStrategy(Product.RAINFOREST_RESIN, PARAMS[Product.RAINFOREST_RESIN]),
            Product.KELP: KelpStrategy(Product.KELP, PARAMS[Product.KELP]),
            Product.SQUID_INK: SquidInkStrategy(Product.SQUID_INK, PARAMS[Product.SQUID_INK]),
            Product.JAMS: KelpStrategy(Product.JAMS, PARAMS[Product.JAMS]),
            Product.CROISSANTS: ComponentStrategy(Product.CROISSANTS, PARAMS[Product.CROISSANTS]),
            Product.DJEMBES: ComponentStrategy(Product.DJEMBES, PARAMS[Product.DJEMBES]),
            Product.VOLCANIC_ROCK_VOUCHER_9500: OptionVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_9500, PARAMS[Product.VOLCANIC_ROCK_VOUCHER_9500]),
            Product.VOLCANIC_ROCK_VOUCHER_9750: OptionVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_9750, PARAMS[Product.VOLCANIC_ROCK_VOUCHER_9750]),
            Product.VOLCANIC_ROCK_VOUCHER_10000: OptionVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10000, PARAMS[Product.VOLCANIC_ROCK_VOUCHER_10000]),
            Product.VOLCANIC_ROCK_VOUCHER_10250: OptionVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10250, PARAMS[Product.VOLCANIC_ROCK_VOUCHER_10250]),
            Product.VOLCANIC_ROCK_VOUCHER_10500: OptionVoucherStrategy(Product.VOLCANIC_ROCK_VOUCHER_10500, PARAMS[Product.VOLCANIC_ROCK_VOUCHER_10500]),
        }
        self.trader_data = {'trade_count': 0, 'last_mid_prices': {}, 'iv_history': []}
        self.macaron_trader = MacaronTrader()

    def run(self, state: TradingState):
        start_time = time.time()
        logger.print("Running Trader code version: 2025-04-17-fix-price-analyzer")
        if state.traderData:
            try:
                self.trader_data = jsonpickle.decode(state.traderData)
            except Exception as e:
                logger.print(f"Error loading saved state: {e}")

        self.price_analyzer.update_fair_values(state.order_depths)
        self.price_analyzer.calculate_basket_values(self.basket_arbitrage.components)
        self.price_analyzer.update_spread_history(list(self.basket_arbitrage.components.keys()))

        priority_products = [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK,
                            Product.VOLCANIC_ROCK_VOUCHER_9500, Product.VOLCANIC_ROCK_VOUCHER_9750]
        result = defaultdict(list)
        conversions = 0
        state.strategies = self.strategies

        for product in priority_products:
            if product not in state.order_depths:
                continue
            strategy = self.strategies[product]
            volatility = np.std(self.price_analyzer.price_history.get(product, [-1])[-10:]) if len(self.price_analyzer.price_history.get(product, [])) >= 10 else 1
            if volatility > 3:
                logger.print(f"High volatility for {product}, reducing trading")
            if isinstance(strategy, ComponentStrategy):
                result[product] = strategy.generate_orders(state, state.order_depths[product], self.price_analyzer)
            else:
                position = state.position.get(product, 0)
                strategy.update_position_window(position)
                soft_liquidate, hard_liquidate = strategy.should_liquidate()
                fair_value = strategy.calculate_fair_value(state, self.price_analyzer)
                if fair_value is None:
                    continue
                stop_loss_orders = strategy.check_stop_loss(state.order_depths[product], fair_value, position)
                result[product].extend(stop_loss_orders)
                take_orders, to_buy, to_sell = strategy.take_orders(state.order_depths[product], fair_value, position)
                clear_orders, to_buy, to_sell = strategy.clear_orders(state.order_depths[product], fair_value, position, to_buy, to_sell)
                make_orders = strategy.make_orders(state.order_depths[product], fair_value, position, to_buy, to_sell, soft_liquidate, hard_liquidate)
                result[product].extend(take_orders + clear_orders + make_orders)

        for product in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if product in state.order_depths:
                result[product].extend(self.basket_arbitrage.generate_orders(state, product))
        if "MAGNIFICENT_MACARONS" in state.order_depths:
                    # Generate orders
                    orders = self.macaron_trader.generate_orders(state)
                    if orders:
                        result["MAGNIFICENT_MACARONS"] = orders
                        
        conversions = self.macaron_trader.determine_conversions(state)

        options_result, options_conversions = self.options_strategy.generate_orders(state, self.price_analyzer)
        for product, orders in options_result.items():
            result[product].extend(orders)
        conversions += options_conversions

        

        actual_positions = state.position.copy()
        for product, trades in state.own_trades.items():
            for trade in trades:
                qty = trade.quantity if trade.buyer == "SUBMISSION" else -trade.quantity
                actual_positions[product] = actual_positions.get(product, 0) + qty
        for product, orders in result.items():
            total_qty = sum(o.quantity for o in orders)
            current_pos = actual_positions.get(product, 0)
            if abs(current_pos + total_qty) > PARAMS[product].position_limit:
                logger.print(f"Warning: Orders for {product} exceed limit, canceling")
                result[product] = []

        trader_data = jsonpickle.encode({
            'trade_count': self.trader_data.get('trade_count', 0),
            'last_mid_prices': self.trader_data.get('last_mid_prices', {}),
            'iv_history': self.trader_data.get('iv_history', [])[-5:]
        })

        
        logger.print(f"Execution time: {time.time() - start_time:.3f} seconds")
        logger.flush(state, result, conversions, trader_data)
        return dict(result), conversions, trader_data