package com.security.trading.derivatives;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class DerivativesService {

    private static final Logger logger = LogManager.getLogger(DerivativesService.class);

    private final ConcurrentHashMap<String, OptionTrade> trades = new ConcurrentHashMap<>();

    public OptionTrade executeTrade(OptionTrade trade) {
        if (trade.getTradeId() == null) {
            trade.setTradeId("OPT-" + LocalDateTime.now().getYear()
                    + "-" + String.format("%05d", trades.size() + 1));
        }
        if (trade.getTimestamp() == null) {
            trade.setTimestamp(LocalDateTime.now());
        }

        logger.info("[{}] Options trade request: type={}, underlying={}, strike={}, expiry={}, style={}, qty={}, impliedVol={}%",
                trade.getTradeId(), trade.getOptionType(), trade.getUnderlying(),
                trade.getStrikePrice(), trade.getExpirationDate(), trade.getOptionStyle(),
                trade.getQuantity(), String.format("%.2f", trade.getImpliedVolatility() * 100));

        if (trade.getDelta() == 0) {
            calculateGreeks(trade);
        }

        trade.setStatus("EXECUTED");
        trade.setFilledQuantity(trade.getQuantity());
        trade.setAveragePrice(trade.getPrice());
        trades.put(trade.getTradeId(), trade);

        logger.info("[{}] Options trade EXECUTED: delta={}, gamma={}, theta={}, vega={}, rho={}",
                trade.getTradeId(),
                String.format("%.4f", trade.getDelta()),
                String.format("%.6f", trade.getGamma()),
                String.format("%.6f", trade.getTheta()),
                String.format("%.4f", trade.getVega()),
                String.format("%.4f", trade.getRho()));
        return trade;
    }

    public OptionPriceResult priceOption(OptionPriceRequest request) {
        logger.info("Black-Scholes pricing: type={}, underlying={}, S={}, K={}, T=1Y, r={}%, sigma={}%",
                request.getOptionType(), request.getUnderlying(),
                request.getUnderlyingPrice(), request.getStrikePrice(),
                String.format("%.4f", request.getRiskFreeRate() * 100),
                String.format("%.2f", request.getImpliedVolatility() * 100));

        double S = request.getUnderlyingPrice();
        double K = request.getStrikePrice();
        double r = request.getRiskFreeRate();
        double sigma = request.getImpliedVolatility();
        double T = 1.0; // 1 year assumed

        // Black-Scholes d1, d2
        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        logger.debug("B-S intermediate: ln(S/K)={}, (r+σ²/2)*T={}, σ√T={}",
                String.format("%.6f", Math.log(S / K)),
                String.format("%.6f", (r + 0.5 * sigma * sigma) * T),
                String.format("%.6f", sigma * Math.sqrt(T)));
        logger.debug("B-S d1 = (ln({}/{}) + ({} + 0.5*{}²)*{}) / ({}*√{}) = {}",
                S, K, r, sigma, T, sigma, T, String.format("%.6f", d1));
        logger.debug("B-S d2 = d1 - σ√T = {} - {} = {}",
                String.format("%.6f", d1), String.format("%.6f", sigma * Math.sqrt(T)),
                String.format("%.6f", d2));

        double Nd1 = normalCDF(d1);
        double Nd2 = normalCDF(d2);
        double nd1 = normalPDF(d1);
        logger.debug("B-S N(d1)={}, N(d2)={}, n(d1)={}",
                String.format("%.6f", Nd1), String.format("%.6f", Nd2), String.format("%.6f", nd1));

        double optionPrice;
        double delta;
        double rho;
        if ("CALL".equalsIgnoreCase(request.getOptionType())) {
            optionPrice = S * Nd1 - K * Math.exp(-r * T) * Nd2;
            delta = Nd1;
            rho = K * T * Math.exp(-r * T) * Nd2 / 100.0;
            logger.debug("CALL price = S*N(d1) - K*e^(-rT)*N(d2) = {}*{} - {}*{}*{} = {}",
                    S, String.format("%.6f", Nd1), K,
                    String.format("%.6f", Math.exp(-r * T)), String.format("%.6f", Nd2),
                    String.format("%.4f", optionPrice));
        } else {
            double NnD1 = normalCDF(-d1);
            double NnD2 = normalCDF(-d2);
            optionPrice = K * Math.exp(-r * T) * NnD2 - S * NnD1;
            delta = Nd1 - 1.0;
            rho = -K * T * Math.exp(-r * T) * NnD2 / 100.0;
            logger.debug("PUT price = K*e^(-rT)*N(-d2) - S*N(-d1) = {}*{}*{} - {}*{} = {}",
                    K, String.format("%.6f", Math.exp(-r * T)), String.format("%.6f", NnD2),
                    S, String.format("%.6f", NnD1), String.format("%.4f", optionPrice));
        }

        double gamma = nd1 / (S * sigma * Math.sqrt(T));
        double theta = ((-S * nd1 * sigma / (2.0 * Math.sqrt(T)))
                - r * K * Math.exp(-r * T) * Nd2) / 365.0;
        double vega = S * nd1 * Math.sqrt(T) / 100.0;

        logger.info("Greeks: delta={}, gamma={}, theta={}/day, vega={}/1%vol, rho={}/1%rate",
                String.format("%.4f", delta),
                String.format("%.6f", gamma),
                String.format("%.6f", theta),
                String.format("%.4f", vega),
                String.format("%.4f", rho));

        // Intrinsic and time value
        double intrinsic = Math.max(0, S - K);
        double timeValue = optionPrice - intrinsic;
        logger.info("Price decomposition: optionPrice={}, intrinsicValue={}, timeValue={}",
                String.format("%.4f", optionPrice),
                String.format("%.4f", intrinsic),
                String.format("%.4f", timeValue));

        if (timeValue < 0) {
            logger.warn("Negative time value detected (timeValue={}): deep ITM option, check inputs",
                    String.format("%.4f", timeValue));
        }

        OptionPriceResult result = new OptionPriceResult();
        result.setOptionPrice(optionPrice);
        result.setDelta(delta);
        result.setGamma(gamma);
        result.setTheta(theta);
        result.setVega(vega);
        result.setRho(rho);
        result.setImpliedVolatility(sigma);
        result.setIntrinsicValue(intrinsic);
        result.setTimeValue(timeValue);
        return result;
    }

    private void calculateGreeks(OptionTrade trade) {
        logger.debug("[{}] Computing Greeks for trade (no explicit request): underlying={}, strike={}, IV={}%",
                trade.getTradeId(), trade.getUnderlying(), trade.getStrikePrice(),
                String.format("%.2f", trade.getImpliedVolatility() * 100));

        double S = trade.getPrice() > 0 ? trade.getPrice() : trade.getStrikePrice() * 1.02;
        double K = trade.getStrikePrice();
        double r = 0.045;
        double sigma = trade.getImpliedVolatility() > 0 ? trade.getImpliedVolatility() : 0.20;
        double T = 0.5; // 6-month assumed

        double d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
        double d2 = d1 - sigma * Math.sqrt(T);
        double nd1 = normalPDF(d1);
        double Nd1 = normalCDF(d1);
        double Nd2 = normalCDF(d2);

        trade.setDelta("CALL".equalsIgnoreCase(trade.getOptionType()) ? Nd1 : Nd1 - 1.0);
        trade.setGamma(nd1 / (S * sigma * Math.sqrt(T)));
        trade.setTheta(((-S * nd1 * sigma / (2.0 * Math.sqrt(T))) - r * K * Math.exp(-r * T) * Nd2) / 365.0);
        trade.setVega(S * nd1 * Math.sqrt(T) / 100.0);
        trade.setRho("CALL".equalsIgnoreCase(trade.getOptionType())
                ? K * T * Math.exp(-r * T) * Nd2 / 100.0
                : -K * T * Math.exp(-r * T) * normalCDF(-d2) / 100.0);

        logger.debug("[{}] Computed Greeks: d1={}, d2={}, delta={}, gamma={}, theta={}, vega={}, rho={}",
                trade.getTradeId(), String.format("%.4f", d1), String.format("%.4f", d2),
                String.format("%.4f", trade.getDelta()), String.format("%.6f", trade.getGamma()),
                String.format("%.6f", trade.getTheta()), String.format("%.4f", trade.getVega()),
                String.format("%.4f", trade.getRho()));
    }

    // Standard normal CDF via approximation
    private double normalCDF(double x) {
        double t = 1.0 / (1.0 + 0.2316419 * Math.abs(x));
        double poly = t * (0.319381530 + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        double approx = 1.0 - normalPDF(x) * poly;
        return x >= 0 ? approx : 1.0 - approx;
    }

    // Standard normal PDF
    private double normalPDF(double x) {
        return Math.exp(-0.5 * x * x) / Math.sqrt(2.0 * Math.PI);
    }

    public List<OptionTrade> getTrades() {
        return new ArrayList<>(trades.values());
    }

    public OptionTrade getTrade(String tradeId) {
        return trades.get(tradeId);
    }
}
