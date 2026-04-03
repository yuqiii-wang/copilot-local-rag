package com.security.trading.risk;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class RiskService {

    private static final Logger logger = LogManager.getLogger(RiskService.class);

    // Z-scores for VaR confidence levels
    private static final double Z_99 = 2.3263;
    private static final double Z_95 = 1.6449;
    // Stress shock magnitude
    private static final double STRESS_SHOCK = 0.15;
    // Position limit (USD)
    private static final double MAX_POSITION_LIMIT = 1_000_000.0;
    // Risk score approval threshold
    private static final double RISK_SCORE_THRESHOLD = 50.0;
    // Assumed annual volatility for VaR (15%)
    private static final double DEFAULT_ANNUAL_VOLATILITY = 0.15;
    // Trading days per year
    private static final int TRADING_DAYS = 252;

    private final ConcurrentHashMap<String, RiskAssessment> assessments = new ConcurrentHashMap<>();

    public RiskAssessment assessRisk(RiskAssessmentRequest request) {
        logger.info("Risk assessment requested: portfolioId={}, value={}, model={}, confidenceLevel={}%, horizon={}d",
                request.getPortfolioId(), String.format("%.2f", request.getPortfolioValue()),
                request.getRiskModel(), request.getConfidenceLevel(), request.getTimeHorizon());

        RiskAssessment assessment = new RiskAssessment();
        assessment.setAssessmentId("RISK-" + LocalDateTime.now().getYear()
                + "-" + String.format("%05d", assessments.size() + 1));
        assessment.setPortfolioId(request.getPortfolioId());
        assessment.setAssessmentTime(LocalDateTime.now());

        double P = request.getPortfolioValue();
        double sigma = DEFAULT_ANNUAL_VOLATILITY;
        int T = request.getTimeHorizon();
        double dailySigma = sigma / Math.sqrt(TRADING_DAYS);
        double horizonSigma = dailySigma * Math.sqrt(T);

        logger.debug("VaR parameters: P={}, annualVol={}%, dailyVol={}%, horizonVol={}%, T={}d",
                String.format("%.2f", P),
                String.format("%.4f", sigma * 100),
                String.format("%.6f", dailySigma * 100),
                String.format("%.6f", horizonSigma * 100), T);

        // Parametric VaR: VaR = Z * sigma_horizon * P
        double var99 = Z_99 * horizonSigma * P;
        double var95 = Z_95 * horizonSigma * P;
        assessment.setVar99(var99);
        assessment.setVar95(var95);
        logger.info("VaR calculation: VaR99 = {} * {} * {} = {} | VaR95 = {} * {} * {} = {}",
                String.format("%.4f", Z_99), String.format("%.6f", horizonSigma), String.format("%.2f", P),
                String.format("%.2f", var99),
                String.format("%.4f", Z_95), String.format("%.6f", horizonSigma), String.format("%.2f", P),
                String.format("%.2f", var95));

        // Stress test: apply STRESS_SHOCK to portfolio value
        double stressLoss = P * STRESS_SHOCK;
        assessment.setStressTestResult(stressLoss);
        logger.info("Stress test: shockMagnitude={}%, portfolioValue={}, stressLoss={}",
                String.format("%.0f", STRESS_SHOCK * 100),
                String.format("%.2f", P),
                String.format("%.2f", stressLoss));

        // Counterparty risk (simplified: 2% of notional)
        double cpRisk = P * 0.02;
        assessment.setCounterpartyRisk(cpRisk);
        logger.debug("Counterparty risk estimate: {}% of notional = {}",
                "2.00", String.format("%.2f", cpRisk));

        // Limit breach check: VaR99 should not exceed 6% of portfolio
        double var99Pct = var99 / P * 100.0;
        double limitPct = 6.0;
        if (var99Pct > limitPct) {
            assessment.setLimitBreached(true);
            assessment.setBreachReason(String.format(
                    "VaR99 (%.2f%%) exceeds portfolio limit (%.2f%%)", var99Pct, limitPct));
            logger.warn("LIMIT BREACH: portfolioId={}, VaR99={}% > limit={}%, breach reason: {}",
                    request.getPortfolioId(), String.format("%.2f", var99Pct),
                    String.format("%.2f", limitPct), assessment.getBreachReason());
        } else {
            assessment.setLimitBreached(false);
            logger.info("Risk limits OK: portfolioId={}, VaR99={}% within limit={}%",
                    request.getPortfolioId(), String.format("%.2f", var99Pct),
                    String.format("%.2f", limitPct));
        }

        assessments.put(assessment.getAssessmentId(), assessment);
        logger.info("Assessment stored: assessmentId={}", assessment.getAssessmentId());
        return assessment;
    }

    public RiskCheckResult checkRisk(RiskCheckRequest request) {
        logger.info("Pre-trade risk check: tradeId={}, tradeType={}, tradeValue={}, qty={}, price={}, counterparty={}",
                request.getTradeId(), request.getTradeType(),
                String.format("%.2f", request.getTradeValue()),
                request.getQuantity(), String.format("%.4f", request.getPrice()),
                request.getCounterparty());

        RiskCheckResult result = new RiskCheckResult();
        result.setTradeId(request.getTradeId());
        result.setMaxPositionLimit(MAX_POSITION_LIMIT);
        result.setCurrentPosition(request.getPositionSize());

        // Risk score = (tradeValue / 10000) + market impact component
        double marketImpactScore = computeMarketImpact(request.getTradeValue(), request.getQuantity());
        double baseRiskScore = request.getTradeValue() / 10_000.0;
        double riskScore = baseRiskScore + marketImpactScore;
        result.setRiskScore(riskScore);

        logger.debug("Risk score: baseScore = tradeValue({}) / 10000 = {}, marketImpact = {}, totalRiskScore = {}",
                String.format("%.2f", request.getTradeValue()),
                String.format("%.4f", baseRiskScore),
                String.format("%.4f", marketImpactScore),
                String.format("%.4f", riskScore));

        // Position limit check
        boolean positionOk = request.getPositionSize() < MAX_POSITION_LIMIT;
        boolean scoreOk = riskScore < RISK_SCORE_THRESHOLD;

        logger.debug("Position limit check: currentPosition={} < maxLimit={} → {}",
                String.format("%.2f", request.getPositionSize()),
                String.format("%.2f", MAX_POSITION_LIMIT),
                positionOk ? "PASS" : "FAIL");
        logger.debug("Risk score check: riskScore={} < threshold={} → {}",
                String.format("%.4f", riskScore), RISK_SCORE_THRESHOLD,
                scoreOk ? "PASS" : "FAIL");

        if (scoreOk && positionOk) {
            result.setApproved(true);
            result.setRiskStatus("APPROVED");
            logger.info("Pre-trade risk check APPROVED: tradeId={}, riskScore={}, positionSize={}",
                    request.getTradeId(), String.format("%.4f", riskScore),
                    String.format("%.2f", request.getPositionSize()));
        } else {
            result.setApproved(false);
            result.setRiskStatus("REJECTED");
            String reason = !positionOk
                    ? String.format("Position size %.2f exceeds limit %.2f",
                            request.getPositionSize(), MAX_POSITION_LIMIT)
                    : String.format("Risk score %.4f exceeds threshold %.4f",
                            riskScore, RISK_SCORE_THRESHOLD);
            result.setRejectionReason(reason);
            logger.warn("Pre-trade risk check REJECTED: tradeId={}, reason={}",
                    request.getTradeId(), reason);
        }

        return result;
    }

    // Simplified market impact: alpha * sqrt(Q / ADV), alpha=0.005, ADV=10000
    private double computeMarketImpact(double tradeValue, int quantity) {
        double alpha = 0.005;
        double adv = 10_000.0;
        double impact = alpha * Math.sqrt((double) quantity / adv) * tradeValue / 1_000.0;
        logger.debug("Market impact: alpha={}, qty={}, ADV={}, impact = alpha * sqrt(qty/ADV) * value/1000 = {}",
                alpha, quantity, adv, String.format("%.6f", impact));
        return impact;
    }

    public List<RiskAssessment> getAssessments() {
        return new ArrayList<>(assessments.values());
    }

    public RiskAssessment getAssessment(String assessmentId) {
        return assessments.get(assessmentId);
    }
}
