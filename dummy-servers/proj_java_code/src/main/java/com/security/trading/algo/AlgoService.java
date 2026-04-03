package com.security.trading.algo;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class AlgoService {

    private static final Logger logger = LogManager.getLogger(AlgoService.class);

    private static final double RISK_FREE_RATE = 0.045; // 4.5%
    private static final int VWAP_SLICES = 12; // per half-hour slice in 6-hour session

    private final ConcurrentHashMap<String, AlgoStrategy> strategies = new ConcurrentHashMap<>();

    public AlgoStrategy createStrategy(AlgoStrategy strategy) {
        if (strategy.getStrategyId() == null) {
            strategy.setStrategyId("ALGO-" + String.format("%05d", strategies.size() + 1));
        }
        if (strategy.getStatus() == null) {
            strategy.setStatus("INACTIVE");
        }
        strategies.put(strategy.getStrategyId(), strategy);
        logger.info("Strategy created: id={}, name={}, type={}, status={}",
                strategy.getStrategyId(), strategy.getName(), strategy.getType(), strategy.getStatus());
        return strategy;
    }

    public AlgoStrategy activateStrategy(String strategyId) {
        AlgoStrategy strategy = strategies.get(strategyId);
        if (strategy != null) {
            strategy.setStatus("ACTIVE");
            logger.info("Strategy ACTIVATED: id={}, type={}", strategyId, strategy.getType());
        } else {
            logger.warn("Activate request for unknown strategyId={}", strategyId);
        }
        return strategy;
    }

    public AlgoStrategy deactivateStrategy(String strategyId) {
        AlgoStrategy strategy = strategies.get(strategyId);
        if (strategy != null) {
            strategy.setStatus("INACTIVE");
            logger.info("Strategy DEACTIVATED: id={}", strategyId);
        } else {
            logger.warn("Deactivate request for unknown strategyId={}", strategyId);
        }
        return strategy;
    }

    public List<AlgoStrategy> getStrategies() {
        return new ArrayList<>(strategies.values());
    }

    public AlgoStrategy getStrategy(String strategyId) {
        return strategies.get(strategyId);
    }

    public BacktestResult backtestStrategy(BacktestRequest request) {
        logger.info("Backtest started: strategyId={}, symbol={}, period={} to {}, initialCapital={}",
                request.getStrategyId(), request.getSymbol(),
                request.getStartDate(), request.getEndDate(), request.getInitialCapital());

        BacktestResult result = new BacktestResult();
        result.setBacktestId("BT-" + String.format("%05d", 1));
        result.setStrategyId(request.getStrategyId());

        // Simulate daily returns stream
        int tradingDays = 252;
        double[] dailyReturns = generateDailyReturns(tradingDays);
        logger.debug("Generated {} simulated daily returns for strategy {}",
                tradingDays, request.getStrategyId());

        // Total return and final capital
        double totalReturnPct = computeTotalReturn(dailyReturns);
        result.setTotalReturn(totalReturnPct);
        result.setFinalCapital(request.getInitialCapital() * (1.0 + totalReturnPct / 100.0));
        logger.info("Backtest total return: {}%, finalCapital={}",
                String.format("%.4f", totalReturnPct),
                String.format("%.2f", result.getFinalCapital()));

        // Sharpe ratio: (R_p - R_f) / σ_p * sqrt(252)
        double meanDailyReturn = mean(dailyReturns);
        double stdDailyReturn = stdDev(dailyReturns, meanDailyReturn);
        double dailyRf = RISK_FREE_RATE / tradingDays;
        double sharpe = (meanDailyReturn - dailyRf) / stdDailyReturn * Math.sqrt(tradingDays);
        result.setSharpeRatio(sharpe);
        logger.info("Sharpe ratio: (meanDailyReturn({}) - dailyRf({})) / stdDev({}) * √252 = {}",
                String.format("%.6f", meanDailyReturn),
                String.format("%.6f", dailyRf),
                String.format("%.6f", stdDailyReturn),
                String.format("%.4f", sharpe));

        if (sharpe < 1.0) {
            logger.warn("Sharpe ratio ({}) below acceptable threshold of 1.0 — strategy may underperform",
                    String.format("%.4f", sharpe));
        }

        // Max drawdown
        double maxDrawdown = computeMaxDrawdown(dailyReturns);
        result.setMaxDrawdown(maxDrawdown * 100.0);
        logger.info("Max drawdown: {}%", String.format("%.4f", maxDrawdown * 100));

        // VWAP execution quality
        double vwap = computeVwap(request);
        logger.info("VWAP benchmark computed over {} slices: VWAP={}",
                VWAP_SLICES, String.format("%.4f", vwap));

        // Win rate and avg win/loss
        int wins = 0;
        double sumWins = 0;
        double sumLoss = 0;
        for (double r : dailyReturns) {
            if (r > 0) { wins++; sumWins += r; }
            else { sumLoss += Math.abs(r); }
        }
        int losses = tradingDays - wins;
        double winRate = (double) wins / tradingDays;
        double avgWin = wins > 0 ? sumWins / wins * 100 : 0;
        double avgLoss = losses > 0 ? sumLoss / losses * 100 : 0;
        result.setTotalTrades(tradingDays);
        result.setWinRate(winRate);
        result.setAverageWin(avgWin);
        result.setAverageLoss(avgLoss);

        logger.info("Win/Loss: wins={}, losses={}, winRate={}%, avgWin={}%, avgLoss={}%",
                wins, losses,
                String.format("%.2f", winRate * 100),
                String.format("%.4f", avgWin),
                String.format("%.4f", avgLoss));

        // Profit factor check
        double profitFactor = sumLoss > 0 ? sumWins / sumLoss : Double.MAX_VALUE;
        if (profitFactor < 1.0) {
            logger.warn("Profit factor ({}) < 1.0: strategy loses more than it gains on average",
                    String.format("%.4f", profitFactor));
        } else {
            logger.info("Profit factor = {} — strategy profitable on average", String.format("%.4f", profitFactor));
        }

        logger.info("Backtest complete: id={}, strategyId={}", result.getBacktestId(), result.getStrategyId());
        return result;
    }

    private double[] generateDailyReturns(int days) {
        double[] returns = new double[days];
        double drift = 0.0003; // ~7.5% annual
        double vol = 0.012;    // ~19% annual vol
        for (int i = 0; i < days; i++) {
            returns[i] = drift + vol * (Math.random() * 2 - 1);
        }
        return returns;
    }

    private double computeTotalReturn(double[] dailyReturns) {
        double cumulative = 1.0;
        for (double r : dailyReturns) cumulative *= (1.0 + r);
        return (cumulative - 1.0) * 100.0;
    }

    private double mean(double[] arr) {
        double s = 0;
        for (double v : arr) s += v;
        return s / arr.length;
    }

    private double stdDev(double[] arr, double mean) {
        double s = 0;
        for (double v : arr) s += (v - mean) * (v - mean);
        return Math.sqrt(s / arr.length);
    }

    private double computeMaxDrawdown(double[] dailyReturns) {
        double peak = 1.0, cumulative = 1.0, maxDD = 0.0;
        for (double r : dailyReturns) {
            cumulative *= (1.0 + r);
            if (cumulative > peak) peak = cumulative;
            double dd = (peak - cumulative) / peak;
            if (dd > maxDD) maxDD = dd;
        }
        return maxDD;
    }

    // Simulate VWAP: Σ(P_i * V_i) / Σ(V_i) over VWAP_SLICES
    private double computeVwap(BacktestRequest request) {
        double basePrice = 150.0; // hypothetical
        double sumPV = 0, sumV = 0;
        for (int i = 0; i < VWAP_SLICES; i++) {
            double slicePrice = basePrice + (Math.random() * 2 - 1);
            double sliceVolume = 10_000 + Math.random() * 5_000;
            sumPV += slicePrice * sliceVolume;
            sumV += sliceVolume;
            logger.debug("VWAP slice {}: price={}, volume={}, cumVWAP={}",
                    i + 1, String.format("%.4f", slicePrice),
                    String.format("%.0f", sliceVolume),
                    String.format("%.4f", sumPV / sumV));
        }
        return sumPV / sumV;
    }
}
