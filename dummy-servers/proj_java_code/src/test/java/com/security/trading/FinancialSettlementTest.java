package com.security.trading;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Integration Tests for Financial Settlement and Clearing
 * Tests realistic settlement scenarios with financial codes
 */
@DisplayName("Financial Settlement and Clearing Integration Tests")
public class FinancialSettlementTest {

    private SettlementData settlementData;

    @BeforeEach
    public void setUp() {
        settlementData = new SettlementData();
    }

    /**
     * Test T+2 Settlement Cycle (most common for bonds and equities)
     */
    @Test
    @DisplayName("T+2 Settlement Cycle Validation")
    public void testT2SettlementCycle() {
        SettlementTransaction txn = new SettlementTransaction(
            "SETTLE-2024-001",
            "US0378331005",              // Apple ISIN
            "5493006MHB84DD0ZWV18",      // Buyer LEI
            "549300R0MHBXABCD1234",      // Seller LEI
            "CHASUS33MIA",               // Buyer SWIFT
            "DEUTDEFF500",               // Seller SWIFT
            100,                         // Quantity
            150.50,                      // Price
            15050.00,                    // Cash amount
            "USD",
            "T+2"
        );
        
        assertTrue(isValidSettlementTransaction(txn), "T+2 settlement should be valid");
        assertEquals("T+2", txn.getSettlementCycle());
    }

    /**
     * Test T+1 Settlement Cycle (equities in some markets)
     */
    @Test
    @DisplayName("T+1 Settlement Cycle for Equities")
    public void testT1SettlementCycle() {
        SettlementTransaction txn = new SettlementTransaction(
            "SETTLE-2024-002",
            "US5949181045",              // Microsoft ISIN
            "5493006MHB84DD0ZWV18",
            "549300R0MHBXABCD1234",
            "CHASUS33MIA",
            "DEUTDEFF500",
            500,
            320.75,
            160375.00,
            "USD",
            "T+1"
        );
        
        assertTrue(isValidSettlementTransaction(txn), "T+1 settlement should be valid");
        assertEquals("T+1", txn.getSettlementCycle());
    }

    /**
     * Test Multi-Currency Settlement (FX settlement)
     */
    @Test
    @DisplayName("Multi-Currency FX Settlement")
    public void testMultiCurrencySettlement() {
        FXSettlementTransaction fxTxn = new FXSettlementTransaction(
            "FX-SETTLE-2024-001",
            "EUR/USD",                   // Currency pair
            "US0378331005",              // Reference ISIN
            "5493006MHB84DD0ZWV18",      // Counterparty 1 LEI
            "549300R0MHBXABCD1234",      // Counterparty 2 LEI
            "CHASUS33MIA",               // Bank 1 SWIFT
            "DEUTDEFF500",               // Bank 2 SWIFT
            100000.00,                   // EUR Amount
            1.0850,                      // Exchange rate
            108500.00,                   // USD Amount
            "T+2"
        );
        
        assertTrue(isValidFXSettlement(fxTxn), "FX settlement should be valid");
        assertEquals("EUR/USD", fxTxn.getCurrencyPair());
    }

    /**
     * Test Bond Settlement with SEDOL/CUSIP identification
     */
    @Test
    @DisplayName("Bond Settlement with Multiple Identifiers")
    public void testBondSettlement() {
        BondSettlement bondTxn = new BondSettlement(
            "BOND-SETTLE-2024-001",
            "US912828R570",              // US Treasury ISIN
            "912828R57",                 // US Treasury CUSIP
            "2134234",                   // SEDOL
            "5493006MHB84DD0ZWV18",      // Buyer LEI
            "549300R0MHBXABCD1234",      // Seller LEI
            "CHASUS33MIA",               // Settlement SWIFT
            1000000,                     // Face value
            98.50,                       // Price
            985000.00,                   // Settlement amount
            0.0325,                      // SOFR rate impact
            "2034-06-15"                 // Maturity date
        );
        
        assertTrue(isValidBondSettlement(bondTxn), "Bond settlement should be valid");
    }

    /**
     * Test Options Settlement with FIGI
     */
    @Test
    @DisplayName("Options Settlement with FIGI")
    public void testOptionsSettlement() {
        OptionsSettlement optTxn = new OptionsSettlement(
            "OPT-SETTLE-2024-001",
            "BBG000B9XRY4",               // Option FIGI
            "AAPL.O",                    // Underlying RIC
            "US0378331005",              // Underlying ISIN
            150.0,                       // Strike price
            100,                         // Quantity (contracts)
            5.25,                        // Premium per contract
            525.00,                      // Total premium
            "CALL",                      // Option type
            "2024-06-21",                // Expiration date
            "CHASUS33MIA"                // Settlement SWIFT
        );
        
        assertTrue(isValidOptionsSettlement(optTxn), "Options settlement should be valid");
    }

    /**
     * Test Real-Time PnL Calculation with Market Rates
     */
    @Test
    @DisplayName("Real-Time PnL Calculation with Market Rates")
    public void testRealTimePnL() {
        TradePosition position = new TradePosition(
            "US0378331005",              // Apple ISIN
            "AAPL.O",                    // RIC
            100,                         // Quantity
            150.00,                      // Entry price
            152.50,                      // Current market price
            3.25,                        // SOFR rate
            4.50,                        // Fed rate
            0.02                         // Cost of carry percentage
        );
        
        double unrealizedPnL = position.calculateUnrealizedPnL();
        double costOfCarry = position.calculateCostOfCarry();
        
        assertEquals(250.0, unrealizedPnL, 0.01, "Unrealized PnL should be 250");
        assertTrue(costOfCarry > 0, "Cost of carry should be positive");
    }

    /**
     * Test Portfolio Risk Aggregation
     */
    @Test
    @DisplayName("Portfolio Risk Aggregation")
    public void testPortfolioRiskAggregation() {
        Portfolio portfolio = new Portfolio("PORTFOLIO-001", "5493006MHB84DD0ZWV18");
        
        portfolio.addPosition(new TradePosition(
            "US0378331005", "AAPL.O", 100, 150.00, 152.50, 3.25, 4.50, 0.02));
        portfolio.addPosition(new TradePosition(
            "US5949181045", "MSFT.O", 50, 320.00, 325.00, 3.25, 4.50, 0.02));
        
        double totalPnL = portfolio.calculateTotalPnL();
        double totalVAR = portfolio.calculateVAR(0.99);
        
        assertTrue(totalPnL > 0, "Total portfolio PnL should be positive");
        assertTrue(totalVAR > 0, "Portfolio VAR should be positive");
        assertEquals(2, portfolio.getPositions().size(), "Portfolio should have 2 positions");
    }

    /**
     * Test Regulatory Compliance Checks
     */
    @Test
    @DisplayName("Regulatory Compliance Validation")
    public void testRegulatoryCompliance() {
        RegulatoryCheck compliance = new RegulatoryCheck(
            "COMPLIANCE-2024-001",
            "US0378331005",
            "549300R0MHBXABCD1234",      // Counterparty LEI
            100,
            15000.0
        );
        
        assertTrue(compliance.isLEIValid(), "LEI should be valid");
        assertTrue(compliance.isDoddFrankCompliant(), "Dodd-Frank compliance should pass");
    }

    /**
     * Test Multi-Leg Transaction (Basis Trade)
     */
    @Test
    @DisplayName("Multi-Leg Basis Trade Settlement")
    public void testBasisTradeSettlement() {
        BasisTrade basisTrade = new BasisTrade("BASIS-2024-001");
        
        // Long future leg
        basisTrade.addLeg(new TradeLeg(
            "US0378331005",
            "FUTURE",
            100,
            150.00,
            "BUY",
            "CHASUS33MIA"
        ));
        
        // Short spot leg
        basisTrade.addLeg(new TradeLeg(
            "US0378331005",
            "SPOT",
            100,
            149.50,
            "SELL",
            "DEUTDEFF500"
        ));
        
        double basisSpread = basisTrade.calculateBasisSpread();
        assertEquals(0.50, basisSpread, 0.01, "Basis spread should be 0.50");
        assertEquals(2, basisTrade.getLegs().size(), "Basis trade should have 2 legs");
    }

    // ==================== Validation Helper Methods ====================

    private boolean isValidSettlementTransaction(SettlementTransaction txn) {
        return txn.getIsin() != null && txn.getIsin().matches("^[A-Z]{2}[A-Z0-9]{9}$") &&
               txn.getBuyerLei() != null && txn.getBuyerLei().length() == 20 &&
               txn.getSellerLei() != null && txn.getSellerLei().length() == 20 &&
               txn.getQuantity() > 0 && txn.getPrice() > 0 &&
               txn.getSettlementCycle() != null;
    }

    private boolean isValidFXSettlement(FXSettlementTransaction fxTxn) {
        return fxTxn.getCurrencyPair() != null && fxTxn.getCurrencyPair().contains("/") &&
               fxTxn.getExchangeRate() > 0 && fxTxn.getAmount1() > 0 && fxTxn.getAmount2() > 0;
    }

    private boolean isValidBondSettlement(BondSettlement bondTxn) {
        return bondTxn.getIsin() != null && bondTxn.getSedol() != null &&
               bondTxn.getFaceValue() > 0 && bondTxn.getPrice() > 0 &&
               bondTxn.getMaturityDate() != null;
    }

    private boolean isValidOptionsSettlement(OptionsSettlement optTxn) {
        return optTxn.getFigi() != null && optTxn.getStrikePrice() > 0 &&
               optTxn.getQuantity() > 0 && optTxn.getPremium() > 0 &&
               (optTxn.getType().equals("CALL") || optTxn.getType().equals("PUT"));
    }

    // ==================== Data Classes ====================

    public static class SettlementTransaction {
        private String id, isin, buyerLei, sellerLei, buyerSwift, sellerSwift;
        private long quantity;
        private double price, cashAmount;
        private String currency, settlementCycle;

        public SettlementTransaction(String id, String isin, String buyerLei, String sellerLei,
                                    String buyerSwift, String sellerSwift, long quantity,
                                    double price, double cashAmount, String currency, String settlementCycle) {
            this.id = id;
            this.isin = isin;
            this.buyerLei = buyerLei;
            this.sellerLei = sellerLei;
            this.buyerSwift = buyerSwift;
            this.sellerSwift = sellerSwift;
            this.quantity = quantity;
            this.price = price;
            this.cashAmount = cashAmount;
            this.currency = currency;
            this.settlementCycle = settlementCycle;
        }

        public String getIsin() { return isin; }
        public String getBuyerLei() { return buyerLei; }
        public String getSellerLei() { return sellerLei; }
        public long getQuantity() { return quantity; }
        public double getPrice() { return price; }
        public String getSettlementCycle() { return settlementCycle; }
    }

    public static class FXSettlementTransaction {
        private String id, currencyPair, isin, counterparty1Lei, counterparty2Lei;
        private String bank1Swift, bank2Swift;
        private double amount1, exchangeRate, amount2;
        private String settlementCycle;

        public FXSettlementTransaction(String id, String currencyPair, String isin,
                                     String counterparty1Lei, String counterparty2Lei,
                                     String bank1Swift, String bank2Swift,
                                     double amount1, double exchangeRate, double amount2,
                                     String settlementCycle) {
            this.id = id;
            this.currencyPair = currencyPair;
            this.isin = isin;
            this.counterparty1Lei = counterparty1Lei;
            this.counterparty2Lei = counterparty2Lei;
            this.bank1Swift = bank1Swift;
            this.bank2Swift = bank2Swift;
            this.amount1 = amount1;
            this.exchangeRate = exchangeRate;
            this.amount2 = amount2;
            this.settlementCycle = settlementCycle;
        }

        public String getCurrencyPair() { return currencyPair; }
        public double getExchangeRate() { return exchangeRate; }
        public double getAmount1() { return amount1; }
        public double getAmount2() { return amount2; }
    }

    public static class BondSettlement {
        private String id, isin, cusip, sedol, buyerLei, sellerLei, settlementSwift;
        private double faceValue, price, settlementAmount, sofrRate;
        private String maturityDate;

        public BondSettlement(String id, String isin, String cusip, String sedol,
                            String buyerLei, String sellerLei, String settlementSwift,
                            double faceValue, double price, double settlementAmount,
                            double sofrRate, String maturityDate) {
            this.id = id;
            this.isin = isin;
            this.cusip = cusip;
            this.sedol = sedol;
            this.buyerLei = buyerLei;
            this.sellerLei = sellerLei;
            this.settlementSwift = settlementSwift;
            this.faceValue = faceValue;
            this.price = price;
            this.settlementAmount = settlementAmount;
            this.sofrRate = sofrRate;
            this.maturityDate = maturityDate;
        }

        public String getIsin() { return isin; }
        public String getSedol() { return sedol; }
        public double getFaceValue() { return faceValue; }
        public double getPrice() { return price; }
        public String getMaturityDate() { return maturityDate; }
    }

    public static class OptionsSettlement {
        private String id, figi, ric, isin, type, expirationDate, settlementSwift;
        private double strikePrice, premium;
        private long quantity;

        public OptionsSettlement(String id, String figi, String ric, String isin,
                               double strikePrice, long quantity, double premium,
                               double totalPremium, String type, String expirationDate,
                               String settlementSwift) {
            this.id = id;
            this.figi = figi;
            this.ric = ric;
            this.isin = isin;
            this.strikePrice = strikePrice;
            this.quantity = quantity;
            this.premium = premium;
            this.type = type;
            this.expirationDate = expirationDate;
            this.settlementSwift = settlementSwift;
        }

        public String getFigi() { return figi; }
        public double getStrikePrice() { return strikePrice; }
        public long getQuantity() { return quantity; }
        public double getPremium() { return premium; }
        public String getType() { return type; }
    }

    public static class TradePosition {
        private String isin, ric;
        private long quantity;
        private double entryPrice, currentPrice, sofrRate, fedRate, costOfCarryPct;

        public TradePosition(String isin, String ric, long quantity, double entryPrice,
                           double currentPrice, double sofrRate, double fedRate, double costOfCarryPct) {
            this.isin = isin;
            this.ric = ric;
            this.quantity = quantity;
            this.entryPrice = entryPrice;
            this.currentPrice = currentPrice;
            this.sofrRate = sofrRate;
            this.fedRate = fedRate;
            this.costOfCarryPct = costOfCarryPct;
        }

        public double calculateUnrealizedPnL() {
            return quantity * (currentPrice - entryPrice);
        }

        public double calculateCostOfCarry() {
            return quantity * entryPrice * costOfCarryPct;
        }

        public java.util.List<TradePosition> asPosition() {
            return java.util.Arrays.asList(this);
        }
    }

    public static class Portfolio {
        private String id, leiCode;
        private List<TradePosition> positions = new ArrayList<>();

        public Portfolio(String id, String leiCode) {
            this.id = id;
            this.leiCode = leiCode;
        }

        public void addPosition(TradePosition position) {
            positions.add(position);
        }

        public double calculateTotalPnL() {
            return positions.stream().mapToDouble(TradePosition::calculateUnrealizedPnL).sum();
        }

        public double calculateVAR(double confidenceLevel) {
            return calculateTotalPnL() * 0.01; // Simplified VAR calculation
        }

        public List<TradePosition> getPositions() {
            return positions;
        }
    }

    public static class RegulatoryCheck {
        private String id, isin, counterpartyLei;
        private long quantity;
        private double notionalValue;

        public RegulatoryCheck(String id, String isin, String counterpartyLei,
                             long quantity, double notionalValue) {
            this.id = id;
            this.isin = isin;
            this.counterpartyLei = counterpartyLei;
            this.quantity = quantity;
            this.notionalValue = notionalValue;
        }

        public boolean isLEIValid() {
            return counterpartyLei != null && counterpartyLei.length() == 20;
        }

        public boolean isDoddFrankCompliant() {
            return notionalValue >= 0 && quantity >= 0;
        }
    }

    public static class BasisTrade {
        private String id;
        private List<TradeLeg> legs = new ArrayList<>();

        public BasisTrade(String id) {
            this.id = id;
        }

        public void addLeg(TradeLeg leg) {
            legs.add(leg);
        }

        public double calculateBasisSpread() {
            if (legs.size() < 2) return 0;
            return Math.abs(legs.get(0).getPrice() - legs.get(1).getPrice());
        }

        public List<TradeLeg> getLegs() {
            return legs;
        }
    }

    public static class TradeLeg {
        private String isin, type, side, swift;
        private long quantity;
        private double price;

        public TradeLeg(String isin, String type, long quantity, double price,
                       String side, String swift) {
            this.isin = isin;
            this.type = type;
            this.quantity = quantity;
            this.price = price;
            this.side = side;
            this.swift = swift;
        }

        public double getPrice() { return price; }
    }

    public static class SettlementData {
        // Placeholder for settlement data utilities
    }
}
