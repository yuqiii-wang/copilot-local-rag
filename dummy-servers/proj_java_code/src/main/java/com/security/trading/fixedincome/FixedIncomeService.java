package com.security.trading.fixedincome;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class FixedIncomeService {

    private static final Logger logger = LogManager.getLogger(FixedIncomeService.class);

    private static final double ACCRUED_INTEREST_RATE = 0.005; // 0.5% per period
    private static final int MAX_YTM_ITERATIONS = 100;
    private static final double YTM_CONVERGENCE_THRESHOLD = 1e-8;

    private final ConcurrentHashMap<String, BondTrade> trades = new ConcurrentHashMap<>();

    public BondTrade executeTrade(BondTrade trade) {
        if (trade.getTradeId() == null) {
            trade.setTradeId("FI-" + LocalDateTime.now().getYear() + "-" + String.format("%05d", trades.size() + 1));
        }
        if (trade.getTimestamp() == null) {
            trade.setTimestamp(LocalDateTime.now());
        }

        logger.info("[{}] Received bond trade request: ISIN={}, side={}, qty={}, price={}, bondType={}",
                trade.getTradeId(), trade.getIsin(), trade.getSide(),
                trade.getQuantity(), trade.getPrice(), trade.getBondType());

        // Pre-trade bond metrics
        logger.debug("[{}] Computing pre-trade metrics: couponRate={}%, maturity={}, YTM_input={}",
                trade.getTradeId(), trade.getCouponRate(), trade.getMaturityDate(), trade.getYieldToMaturity());

        double faceValue = trade.getPrice() * trade.getQuantity();
        double couponCash = faceValue * (trade.getCouponRate() / 100.0);
        logger.debug("[{}] Face value={}, annual coupon cash flow={}", trade.getTradeId(), faceValue, couponCash);

        // YTM check
        if (trade.getYieldToMaturity() <= 0) {
            double estimatedYtm = estimateYtm(trade.getPrice(), faceValue / trade.getQuantity(),
                    trade.getCouponRate(), 10);
            logger.info("[{}] YTM not provided, estimated via Newton-Raphson: YTM={}%",
                    trade.getTradeId(), String.format("%.4f", estimatedYtm * 100));
            trade.setYieldToMaturity(estimatedYtm * 100);
        }

        // Duration check
        if (trade.getDuration() <= 0) {
            double y = trade.getYieldToMaturity() / 100.0;
            double macaulayDuration = computeMacaulayDuration(trade.getPrice(), trade.getCouponRate() / 100.0,
                    y, 10);
            double modifiedDuration = macaulayDuration / (1.0 + y);
            logger.info("[{}] Duration not provided, computed: MacaulayDuration={} years, ModifiedDuration={} years",
                    trade.getTradeId(),
                    String.format("%.4f", macaulayDuration),
                    String.format("%.4f", modifiedDuration));
            trade.setDuration(modifiedDuration);
        }

        // Convexity check
        if (trade.getConvexity() <= 0) {
            double y = trade.getYieldToMaturity() / 100.0;
            double convexity = computeConvexity(trade.getPrice(), trade.getCouponRate() / 100.0, y, 10);
            logger.info("[{}] Convexity computed: convexity={}", trade.getTradeId(),
                    String.format("%.4f", convexity));
            trade.setConvexity(convexity);
        }

        // DV01 (dollar value of a basis point)
        double dv01 = trade.getDuration() * faceValue * 0.0001;
        logger.info("[{}] DV01={} (1bp price sensitivity on notional {})",
                trade.getTradeId(), String.format("%.2f", dv01), String.format("%.2f", faceValue));

        // Execution
        trade.setStatus("EXECUTED");
        trade.setFilledQuantity(trade.getQuantity());
        trade.setAveragePrice(trade.getPrice());
        trades.put(trade.getTradeId(), trade);

        logger.info("[{}] Bond trade EXECUTED: ISIN={}, notional={}, YTM={}%, duration={}, convexity={}, DV01={}",
                trade.getTradeId(), trade.getIsin(),
                String.format("%.2f", faceValue),
                String.format("%.4f", trade.getYieldToMaturity()),
                String.format("%.4f", trade.getDuration()),
                String.format("%.4f", trade.getConvexity()),
                String.format("%.2f", dv01));

        return trade;
    }

    public BondCalculationResult calculateBondMetrics(BondCalculationRequest request) {
        logger.info("Bond metrics calculation requested: ISIN={}, faceValue={}, couponRate={}%, marketPrice={}",
                request.getIsin(), request.getFaceValue(),
                request.getCouponRate(), request.getMarketPrice());

        BondCalculationResult result = new BondCalculationResult();
        result.setIsin(request.getIsin());

        // Accrued interest and dirty price
        double accruedInterest = request.getFaceValue() * ACCRUED_INTEREST_RATE;
        double dirtyPrice = request.getMarketPrice() + accruedInterest;
        result.setCleanPrice(request.getMarketPrice());
        result.setDirtyPrice(dirtyPrice);
        logger.debug("ISIN={}: cleanPrice={}, accruedInterest={}, dirtyPrice={}",
                request.getIsin(),
                String.format("%.4f", request.getMarketPrice()),
                String.format("%.4f", accruedInterest),
                String.format("%.4f", dirtyPrice));

        // Current yield
        double currentYield = (request.getCouponRate() / 100.0 * request.getFaceValue()) / request.getMarketPrice() * 100.0;
        result.setCurrentYield(currentYield);
        logger.debug("ISIN={}: currentYield = (coupon {} * faceValue {}) / marketPrice {} * 100 = {}%",
                request.getIsin(), request.getCouponRate(), request.getFaceValue(),
                request.getMarketPrice(), String.format("%.4f", currentYield));

        // YTM via Newton-Raphson
        double ytm = estimateYtm(request.getMarketPrice(), request.getFaceValue(),
                request.getCouponRate(), 10);
        result.setYieldToMaturity(ytm * 100);
        logger.info("ISIN={}: YTM (Newton-Raphson, 10Y) = {}%",
                request.getIsin(), String.format("%.4f", ytm * 100));

        // Macaulay and Modified Duration
        double macaulay = computeMacaulayDuration(request.getMarketPrice(),
                request.getCouponRate() / 100.0, ytm, 10);
        double modified = macaulay / (1.0 + ytm);
        result.setDuration(modified);
        logger.info("ISIN={}: MacaulayDuration={} years → ModifiedDuration={} years",
                request.getIsin(),
                String.format("%.4f", macaulay),
                String.format("%.4f", modified));

        // Convexity
        double convexity = computeConvexity(request.getMarketPrice(),
                request.getCouponRate() / 100.0, ytm, 10);
        result.setConvexity(convexity);
        logger.info("ISIN={}: Convexity = {}", request.getIsin(), String.format("%.4f", convexity));

        // Spread over benchmark (simulated)
        double benchmarkYield = 0.0425; // 4.25% 10Y benchmark
        double spread = ytm - benchmarkYield;
        logger.info("ISIN={}: SpreadOverBenchmark = {} bps (YTM={}%, 10YBenchmark={}%)",
                request.getIsin(),
                String.format("%.1f", spread * 10000),
                String.format("%.4f", ytm * 100),
                String.format("%.4f", benchmarkYield * 100));

        if (spread < 0) {
            logger.warn("ISIN={}: Bond trading BELOW benchmark yield by {} bps — check pricing",
                    request.getIsin(), String.format("%.1f", Math.abs(spread) * 10000));
        }

        return result;
    }

    // Newton-Raphson YTM solver
    private double estimateYtm(double price, double faceValue, double couponRate, int periods) {
        double coupon = faceValue * couponRate / 100.0;
        double ytm = coupon / price; // initial guess = current yield
        logger.debug("YTM Newton-Raphson: initialGuess={}%, faceValue={}, coupon={}, price={}, periods={}",
                String.format("%.4f", ytm * 100), faceValue, coupon, price, periods);

        for (int i = 0; i < MAX_YTM_ITERATIONS; i++) {
            double fVal = 0.0;
            double fDeriv = 0.0;
            for (int t = 1; t <= periods; t++) {
                double disc = Math.pow(1.0 + ytm, t);
                fVal += coupon / disc;
                fDeriv -= t * coupon / (disc * (1.0 + ytm));
            }
            fVal += faceValue / Math.pow(1.0 + ytm, periods) - price;
            fDeriv -= periods * faceValue / (Math.pow(1.0 + ytm, periods) * (1.0 + ytm));

            double delta = fVal / fDeriv;
            ytm -= delta;
            logger.debug("YTM iteration {}: ytm={}%, f(ytm)={}, delta={}",
                    i + 1, String.format("%.6f", ytm * 100),
                    String.format("%.8f", fVal), String.format("%.10f", delta));
            if (Math.abs(delta) < YTM_CONVERGENCE_THRESHOLD) {
                logger.debug("YTM converged at iteration {} with delta={}", i + 1, delta);
                break;
            }
        }
        return ytm;
    }

    // Macaulay Duration = Σ[ t * PV(CFt) ] / Price
    private double computeMacaulayDuration(double price, double couponRate, double ytm, int periods) {
        double coupon = price * couponRate;
        double weightedSum = 0.0;
        double pvSum = 0.0;
        for (int t = 1; t <= periods; t++) {
            double pv = coupon / Math.pow(1.0 + ytm, t);
            weightedSum += t * pv;
            pvSum += pv;
        }
        double pvFace = price / Math.pow(1.0 + ytm, periods);
        weightedSum += periods * pvFace;
        pvSum += pvFace;
        double macaulay = weightedSum / pvSum;
        logger.debug("MacaulayDuration: Σ(t*PV(CF_t))={}, Σ(PV(CF_t))={}, result={}",
                String.format("%.4f", weightedSum), String.format("%.4f", pvSum),
                String.format("%.4f", macaulay));
        return macaulay;
    }

    // Convexity = Σ[ t*(t+1)*PV(CFt) ] / (Price * (1+ytm)^2)
    private double computeConvexity(double price, double couponRate, double ytm, int periods) {
        double coupon = price * couponRate;
        double sum = 0.0;
        for (int t = 1; t <= periods; t++) {
            sum += (t * (t + 1.0) * coupon) / Math.pow(1.0 + ytm, t + 2);
        }
        sum += (periods * (periods + 1.0) * price) / Math.pow(1.0 + ytm, periods + 2);
        double convexity = sum / price;
        logger.debug("Convexity: Σ[t(t+1)*PV(CF_t)] / (P*(1+y)^2) = {}", String.format("%.4f", convexity));
        return convexity;
    }

    public List<BondTrade> getTrades() {
        return new ArrayList<>(trades.values());
    }

    public BondTrade getTrade(String tradeId) {
        return trades.get(tradeId);
    }
}
