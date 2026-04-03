package com.security.trading.settlement;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class SettlementService {
    
    private static final Logger logger = LogManager.getLogger(SettlementService.class);
    
    private final ConcurrentHashMap<String, Settlement> settlements = new ConcurrentHashMap<>();
    
    public Settlement createSettlement(Settlement settlement) {
        // Generate settlement ID if not provided
        if (settlement.getSettlementId() == null) {
            settlement.setSettlementId("SETTLE-" + LocalDateTime.now().getYear() + "-" + String.format("%05d", settlements.size() + 1));
        }
        
        // Set timestamps
        LocalDateTime now = LocalDateTime.now();
        settlement.setCreatedAt(now);
        settlement.setUpdatedAt(now);
        
        // Set initial status
        if (settlement.getStatus() == null) {
            settlement.setStatus("PENDING");
        }
        
        // Store settlement
        settlements.put(settlement.getSettlementId(), settlement);
        
        return settlement;
    }
    
    public Settlement processSettlement(String settlementId) {
        Settlement settlement = settlements.get(settlementId);
        if (settlement != null) {
            try {
                logger.debug("Processing settlement: settlementId={}, tradeId={}, amount={}", 
                    settlementId, settlement.getTradeId(), settlement.getSettlementAmount());
                
                // Validate security
                String isin = settlement.getIsin();
                if (isin == null || !isin.matches("^[A-Z]{2}[A-Z0-9]{9}[0-9]$")) {
                    logger.error("Settlement FAILED for settlementId={}: settlement instruction failed for ISIN validation", settlementId);
                    settlement.setStatus("FAILED");
                    settlement.setFailureReason("Invalid ISIN");
                    return settlement;
                }
                
                // Check cash adequacy: requiredCash = settlementAmount + settlementFee
                double requiredCash = settlement.getSettlementAmount() * 1.0001; // 0.01% fee
                double availableCash = 500000.0; // Simulated available cash
                
                if (requiredCash > availableCash) {
                    double shortfall = requiredCash - availableCash;
                    logger.error("Settlement instruction FAILED: settlementId={}, reason=INSUFFICIENT_CASH, " +
                        "requiredCash={}, availableCash={}, shortfall={}", 
                        settlementId, String.format("%.2f", requiredCash), 
                        String.format("%.2f", availableCash), String.format("%.2f", shortfall));
                    settlement.setStatus("FAILED");
                    settlement.setFailureReason("Insufficient funds");
                    return settlement;
                }
                
                // Calculate settlement metrics
                double settlementFee = settlement.getSettlementAmount() * 0.0001; // 0.01% fee
                logger.info("Settlement fee calculation: settlementFee = settlementAmount({}) * settlementFeeRate(0.01%) = {} * 0.0001 = {}",
                    String.format("%.2f", settlement.getSettlementAmount()),
                    String.format("%.2f", settlement.getSettlementAmount()),
                    String.format("%.2f", settlementFee));
                
                // Process settlement (90% success rate)
                if (Math.random() > 0.1) {
                    settlement.setStatus("PROCESSED");
                    logger.info("Settlement processed successfully: settlementId={}, amount={}, finalStatus=PROCESSED",
                        settlementId, String.format("%.2f", settlement.getSettlementAmount()));
                } else {
                    settlement.setStatus("FAILED");
                    settlement.setFailureReason("Operational error");
                    logger.warn("Settlement processing failed: settlementId={}, reason={}", 
                        settlementId, settlement.getFailureReason());
                }
            } catch (Exception e) {
                logger.error("Settlement processing exception: settlementId={}, exception={}", 
                    settlementId, e.getMessage(), e);
                settlement.setStatus("FAILED");
                settlement.setFailureReason("Processing error");
            }
            settlement.setUpdatedAt(LocalDateTime.now());
        } else {
            logger.error("Settlement not found: settlementId={}", settlementId);
        }
        return settlement;
    }
    
    /**
     * Batch settlement processing with comprehensive logging
     */
    public void processBatchSettlement(LocalDateTime settlementDate) {
        logger.info("Settlement batch initiated: batchId=SETT-BATCH-{}-001, settlementDate={}", 
            settlementDate.toString().replace("-", ""), settlementDate);
        
        int totalTrades = settlements.size();
        double totalValue = settlements.values().stream()
            .mapToDouble(Settlement::getSettlementAmount)
            .sum();
        
        logger.info("Settlement batch initiated: batchId=SETT-BATCH-{}-001, totalTrades={}, totalValue={}", 
            settlementDate.toString().replace("-", ""), totalTrades, String.format("%.2f", totalValue));
        
        // Validate all settlements
        int validCount = 0;
        int invalidCount = 0;
        List<String> rejectedIds = new ArrayList<>();
        
        for (Settlement s : settlements.values()) {
            if (isValidSettlement(s)) {
                validCount++;
            } else {
                invalidCount++;
                rejectedIds.add(s.getSettlementId());
            }
        }
        
        logger.info("Validating trade instructions: total={}, passed={}, failed={}, rejectionRate={}%",
            totalTrades, validCount, invalidCount, 
            String.format("%.2f", (invalidCount * 100.0 / totalTrades)));
        
        if (!rejectedIds.isEmpty()) {
            logger.warn("Failed settlement instructions: tradeIds={}, causes=[INVALID_COUNTERPARTY, MISSING_SETTLEMENT_ACCT]",
                rejectedIds);
        }
        
        // Cash settlement processing with FX conversion
        double cashUSD = 0, cashEUR = 0, cashGBP = 0;
        for (Settlement s : settlements.values()) {
            if (isValidSettlement(s)) {
                double baseAmount = s.getSettlementAmount();
                // Simulate FX rates
                cashUSD += baseAmount * 1.0;     // USD base
                cashEUR += baseAmount * 0.071;   // EUR conversion (approx 1 EUR = 14 USD)
                cashGBP += baseAmount * 0.059;   // GBP conversion (approx 1 GBP = 17 USD)
            }
        }
        
        logger.info("Cash settlement processing: currencyCash=[{{currency=USD, amount={}}}, {{currency=EUR, amount={}}}, {{currency=GBP, amount={}}}]",
            String.format("%.2f", cashUSD), String.format("%.2f", cashEUR), String.format("%.2f", cashGBP));
        
        logger.debug("Cash calculation formulas: cashUSD = sum(notional * FXRate) for USD settlements = {}, " +
            "cashEUR = sum(...) = {}, cashGBP = sum(...) = {}",
            String.format("%.2f", cashUSD), String.format("%.2f", cashEUR), String.format("%.2f", cashGBP));
        
        // Settlement fee calculation
        double totalFee = totalValue * 0.0001; // 0.01% fee
        logger.info("Settlement fee calculation: settlementFee = totalValue({}) * settlementFeeRate(0.01%) = {} * 0.0001 = {}",
            String.format("%.2f", totalValue), String.format("%.2f", totalValue), String.format("%.2f", totalFee));
    }
    
    private boolean isValidSettlement(Settlement settlement) {
        // Check ISIN format
        String isin = settlement.getIsin();
        if (isin == null || !isin.matches("^[A-Z]{2}[A-Z0-9]{9}[0-9]$")) {
            return false;
        }
        // Check required fields
        return settlement.getSettlementAmount() > 0 &&
               settlement.getTradeId() != null &&
               !settlement.getTradeId().isEmpty();
    }
    
    public List<Settlement> getSettlements() {
        return new ArrayList<>(settlements.values());
    }
    
    public Settlement getSettlement(String settlementId) {
        return settlements.get(settlementId);
    }
    
    public List<Settlement> getSettlementsByTradeId(String tradeId) {
        List<Settlement> result = new ArrayList<>();
        for (Settlement settlement : settlements.values()) {
            if (tradeId.equals(settlement.getTradeId())) {
                result.add(settlement);
            }
        }
        return result;
    }
}
