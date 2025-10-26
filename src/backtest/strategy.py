import pandas as pd
import numpy as np
from datetime import datetime


class TradingStrategy:
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def backtest(self, predictions: pd.Series, actual_prices: pd.Series):
        """
        Backtest trading strategy based on predictions.
        predictions: Series of predicted returns
        actual_prices: Series of actual prices
        """
        signals = pd.DataFrame(index=predictions.index)
        signals['prediction'] = predictions
        signals['price'] = actual_prices
        
        # Generate trading signals
        signals['signal'] = 0
        signals.loc[signals['prediction'] > 0.001, 'signal'] = 1  # Buy
        signals.loc[signals['prediction'] < -0.001, 'signal'] = -1  # Sell
        
        # Calculate positions and returns
        signals['position'] = signals['signal'].shift(1).fillna(0)
        signals['actual_return'] = signals['price'].pct_change()
        signals['strategy_return'] = signals['position'] * signals['actual_return']
        
        # Apply commission
        signals['trades'] = signals['position'].diff().abs()
        signals['commission'] = signals['trades'] * self.commission
        signals['net_return'] = signals['strategy_return'] - signals['commission']
        
        # Calculate cumulative returns
        signals['cumulative'] = (1 + signals['net_return']).cumprod()
        signals['portfolio_value'] = self.initial_capital * signals['cumulative']
        
        # Calculate metrics
        metrics = self.calculate_metrics(signals)
        
        return signals, metrics
    
    def calculate_metrics(self, signals: pd.DataFrame):
        """Calculate trading performance metrics."""
        returns = signals['net_return'].dropna()
        
        total_return = (signals['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        annualized_return = (1 + total_return/100) ** (252/len(returns)) - 1
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        cumulative = signals['cumulative'].fillna(1)
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": int(signals['trades'].sum()),
            "final_value": round(signals['portfolio_value'].iloc[-1], 2)
        }