package com.security.trading;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test suite for Financial Code Validation
 * Tests ISIN, SEDOL, CUSIP, LEI, SWIFT, RIC, FIGI and other financial identifiers
 */
@DisplayName("Financial Code Validation Tests")
public class FinancialCodeValidatorTest {

    /**
     * Test ISIN (International Securities Identification Number) validation
     * Format: 2-letter country code + 9 alphanumeric characters (12 chars total)
     */
    @Test
    @DisplayName("ISIN Code Validation")
    public void testISINValidation() {
        // Valid ISINs
        String appleISIN = "US0378331005";
        String microsoftISIN = "US5949181045";
        
        assertTrue(isValidISIN(appleISIN), "Apple ISIN should be valid");
        assertTrue(isValidISIN(microsoftISIN), "Microsoft ISIN should be valid");
        
        // Invalid ISINs
        assertFalse(isValidISIN("US037833"), "ISIN too short should fail");
        assertFalse(isValidISIN("1234567890123"), "ISIN with invalid format should fail");
        assertFalse(isValidISIN(""), "Empty ISIN should fail");
    }

    /**
     * Test SEDOL (Stock Exchange Daily Official List) validation
     * Format: 6 alphanumeric + 1 check digit (7 chars total)
     */
    @Test
    @DisplayName("SEDOL Code Validation")
    public void testSEDOLValidation() {
        // Valid SEDOLs
        String validSEDOL1 = "0263494";
        String validSEDOL2 = "B0LD5H2";
        
        assertTrue(isValidSEDOL(validSEDOL1), "SEDOL 0263494 should be valid");
        assertTrue(isValidSEDOL(validSEDOL2), "SEDOL B0LD5H2 should be valid");
        
        // Invalid SEDOLs
        assertFalse(isValidSEDOL("026349"), "SEDOL too short should fail");
        assertFalse(isValidSEDOL("0263494999"), "SEDOL too long should fail");
        assertFalse(isValidSEDOL(""), "Empty SEDOL should fail");
    }

    /**
     * Test CUSIP (Committee on Uniform Security Identification Procedures) validation
     * Format: 8 alphanumeric + 1 check digit (9 chars total)
     */
    @Test
    @DisplayName("CUSIP Code Validation")
    public void testCUSIPValidation() {
        // Valid CUSIPs
        String appleCP = "037833100";
        String microsoftCUSIP = "594918104";
        
        assertTrue(isValidCUSIP(appleCP), "Apple CUSIP should be valid");
        assertTrue(isValidCUSIP(microsoftCUSIP), "Microsoft CUSIP should be valid");
        
        // Invalid CUSIPs
        assertFalse(isValidCUSIP("03783310"), "CUSIP too short should fail");
        assertFalse(isValidCUSIP("0378331009999"), "CUSIP too long should fail");
        assertFalse(isValidCUSIP(""), "Empty CUSIP should fail");
    }

    /**
     * Test LEI (Legal Entity Identifier) validation
     * Format: 20 alphanumeric characters
     */
    @Test
    @DisplayName("LEI Code Validation")
    public void testLEIValidation() {
        // Valid LEIs
        String validLEI1 = "5493006MHB84DD0ZWV18";
        String validLEI2 = "549300R0MHBXABCD1234";
        
        assertTrue(isValidLEI(validLEI1), "LEI 5493006MHB84DD0ZWV18 should be valid");
        assertTrue(isValidLEI(validLEI2), "LEI 549300R0MHBXABCD1234 should be valid");
        
        // Invalid LEIs
        assertFalse(isValidLEI("5493006MHB84DD0ZWV"), "LEI too short should fail");
        assertFalse(isValidLEI("5493006MHB84DD0ZWV1899999"), "LEI too long should fail");
        assertFalse(isValidLEI(""), "Empty LEI should fail");
    }

    /**
     * Test SWIFT Code validation
     * Format: 4 letter bank code + 2 letter country + 2 letter location + 3 letter branch (optional)
     * 8 or 11 alphanumeric characters
     */
    @Test
    @DisplayName("SWIFT Code Validation")
    public void testSWIFTValidation() {
        // Valid SWIFT codes
        String jpmSwift = "CHASUS33MIA";
        String deutscheSwift = "DEUTDEFF";
        
        assertTrue(isValidSWIFT(jpmSwift), "SWIFT CHASUS33MIA should be valid");
        assertTrue(isValidSWIFT(deutscheSwift), "SWIFT DEUTDEFF should be valid");
        
        // Invalid SWIFT codes
        assertFalse(isValidSWIFT("CHASUS3"), "SWIFT too short should fail");
        assertFalse(isValidSWIFT("CHASUS33MIAXXXX"), "SWIFT too long should fail");
        assertFalse(isValidSWIFT(""), "Empty SWIFT should fail");
    }

    /**
     * Test RIC Code (Reuters Instrument Code) validation
     * Format: 4-7 letter ticker + . + exchange code
     */
    @Test
    @DisplayName("RIC Code Validation")
    public void testRICValidation() {
        // Valid RICs
        String appleRIC = "AAPL.O";      // NASDAQ
        String btRIC = "BT.L";            // London Stock Exchange
        String dxRIC = "0#DXY:IB";        // Dollar Index
        
        assertTrue(isValidRIC(appleRIC), "RIC AAPL.O should be valid");
        assertTrue(isValidRIC(btRIC), "RIC BT.L should be valid");
        assertTrue(isValidRIC(dxRIC), "RIC 0#DXY:IB should be valid");
        
        // Invalid RICs
        assertFalse(isValidRIC("AAPL"), "RIC without exchange should fail");
        assertFalse(isValidRIC(""), "Empty RIC should fail");
    }

    /**
     * Test FIGI (Financial Instrument Global Identifier) validation
     * Format: BBG + 9 alphanumeric characters (12 chars total)
     */
    @Test
    @DisplayName("FIGI Code Validation")
    public void testFIGIValidation() {
        // Valid FIGIs
        String appleFIGI = "BBG000B9XRY4";
        String testFIGI = "BBG0001234567";
        
        assertTrue(isValidFIGI(appleFIGI), "FIGI BBG000B9XRY4 should be valid");
        assertTrue(isValidFIGI(testFIGI), "FIGI BBG0001234567 should be valid");
        
        // Invalid FIGIs
        assertFalse(isValidFIGI("BBG000B9XRY"), "FIGI too short should fail");
        assertFalse(isValidFIGI("XYZ000B9XRY4"), "FIGI without BBG prefix should fail");
        assertFalse(isValidFIGI(""), "Empty FIGI should fail");
    }

    /**
     * Test Currency Pair validation for FX trading
     * Format: XXX/YYY (3-letter ISO 4217 currency codes)
     */
    @Test
    @DisplayName("Currency Pair Validation")
    public void testCurrencyPairValidation() {
        // Valid pairs
        String eurusd = "EUR/USD";
        String gbpjpy = "GBP/JPY";
        
        assertTrue(isValidCurrencyPair(eurusd), "Currency pair EUR/USD should be valid");
        assertTrue(isValidCurrencyPair(gbpjpy), "Currency pair GBP/JPY should be valid");
        
        // Invalid pairs
        assertFalse(isValidCurrencyPair("EU/USD"), "Invalid currency code should fail");
        assertFalse(isValidCurrencyPair("EUR-USD"), "Wrong separator should fail");
        assertFalse(isValidCurrencyPair(""), "Empty pair should fail");
    }

    /**
     * Test Market Rates
     */
    @Test
    @DisplayName("Market Rates Validation")
    public void testMarketRatesValidation() {
        // Valid rates
        double sofrRate = 3.25;
        double fedRate = 4.50;
        
        assertTrue(isValidSOFRRate(sofrRate), "SOFR rate 3.25% should be valid");
        assertTrue(isValidFedRate(fedRate), "Fed rate 4.50% should be valid");
        
        // Invalid rates
        assertFalse(isValidSOFRRate(-0.5), "Negative SOFR should fail");
        assertFalse(isValidFedRate(15.0), "Fed rate above 10% should fail");
    }

    /**
     * Integration test for complete trade validation
     */
    @Test
    @DisplayName("Complete Trade Data Validation")
    public void testCompleteTradeValidation() {
        TradeData trade = new TradeData(
            "TRADE-100001",
            "US0378331005",      // Apple ISIN
            "037833100",         // Apple CUSIP
            "0263494",           // Sample SEDOL
            "5493006MHB84DD0ZWV18", // Sample LEI
            "CHASUS33MIA",       // JP Morgan SWIFT
            "AAPL.O",            // Apple RIC
            "BBG000B9XRY4",      // Apple FIGI
            "USD",
            100,
            150.50,
            "BUY",
            3.25,
            4.50
        );
        
        assertTrue(isValidTradeData(trade), "Complete trade data should be valid");
    }

    // ==================== Helper Validation Methods ====================

    private boolean isValidISIN(String isin) {
        return isin != null && isin.matches("^[A-Z]{2}[A-Z0-9]{9}$");
    }

    private boolean isValidSEDOL(String sedol) {
        return sedol != null && sedol.matches("^[A-Z0-9]{7}$") && sedol.length() == 7;
    }

    private boolean isValidCUSIP(String cusip) {
        return cusip != null && cusip.matches("^[A-Z0-9]{9}$") && cusip.length() == 9;
    }

    private boolean isValidLEI(String lei) {
        return lei != null && lei.matches("^[A-Z0-9]{20}$") && lei.length() == 20;
    }

    private boolean isValidSWIFT(String swift) {
        return swift != null && swift.matches("^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}[A-Z0-9]{0,3}$") && 
               (swift.length() == 8 || swift.length() == 11);
    }

    private boolean isValidRIC(String ric) {
        return ric != null && ric.contains(".") && ric.split("\\.").length == 2;
    }

    private boolean isValidFIGI(String figi) {
        return figi != null && figi.matches("^BBG[A-Z0-9]{9}$") && figi.length() == 12;
    }

    private boolean isValidCurrencyPair(String pair) {
        if (pair == null || !pair.contains("/")) return false;
        String[] parts = pair.split("/");
        return parts.length == 2 && 
               parts[0].matches("^[A-Z]{3}$") && 
               parts[1].matches("^[A-Z]{3}$");
    }

    private boolean isValidSOFRRate(double rate) {
        return rate >= 0 && rate <= 10;
    }

    private boolean isValidFedRate(double rate) {
        return rate >= 0 && rate <= 10;
    }

    private boolean isValidTradeData(TradeData trade) {
        return isValidISIN(trade.getIsin()) &&
               isValidCUSIP(trade.getCusip()) &&
               isValidSEDOL(trade.getSedol()) &&
               isValidLEI(trade.getLei()) &&
               isValidSWIFT(trade.getSwift()) &&
               isValidRIC(trade.getRic()) &&
               isValidFIGI(trade.getFigi()) &&
               isValidCurrencyPair(trade.getCurrency() + "/USD") &&
               trade.getQuantity() > 0 &&
               trade.getPrice() > 0 &&
               isValidSOFRRate(trade.getSofrRate()) &&
               isValidFedRate(trade.getFedRate());
    }

    /**
     * Inner class for trade data used in integration tests
     */
    public static class TradeData {
        private String tradeId;
        private String isin;
        private String cusip;
        private String sedol;
        private String lei;
        private String swift;
        private String ric;
        private String figi;
        private String currency;
        private long quantity;
        private double price;
        private String side;
        private double sofrRate;
        private double fedRate;

        public TradeData(String tradeId, String isin, String cusip, String sedol, 
                        String lei, String swift, String ric, String figi, 
                        String currency, long quantity, double price, String side,
                        double sofrRate, double fedRate) {
            this.tradeId = tradeId;
            this.isin = isin;
            this.cusip = cusip;
            this.sedol = sedol;
            this.lei = lei;
            this.swift = swift;
            this.ric = ric;
            this.figi = figi;
            this.currency = currency;
            this.quantity = quantity;
            this.price = price;
            this.side = side;
            this.sofrRate = sofrRate;
            this.fedRate = fedRate;
        }

        // Getters
        public String getTradeId() { return tradeId; }
        public String getIsin() { return isin; }
        public String getCusip() { return cusip; }
        public String getSedol() { return sedol; }
        public String getLei() { return lei; }
        public String getSwift() { return swift; }
        public String getRic() { return ric; }
        public String getFigi() { return figi; }
        public String getCurrency() { return currency; }
        public long getQuantity() { return quantity; }
        public double getPrice() { return price; }
        public String getSide() { return side; }
        public double getSofrRate() { return sofrRate; }
        public double getFedRate() { return fedRate; }
    }
}
