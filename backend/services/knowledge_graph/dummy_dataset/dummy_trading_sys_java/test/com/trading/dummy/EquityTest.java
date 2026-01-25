package com.trading.dummy;

import com.trading.dummy.model.Side;
import org.junit.Test;

public class EquityTest extends DummyTestBase {

    @Override
    public void setUp() {
        super.setUp();
        // Load Equities
        createEquity("AAPL", "US0378331005", "NASDAQ", 150.00, 150.10, 150.05);
        createEquity("MSFT", "US5949181045", "NASDAQ", 300.00, 300.20, 300.10);
        createEquity("GOOGL", "US02079K3059", "NASDAQ", 2800.00, 2805.00, 2802.50);
        createEquity("AMZN", "US0231351067", "NASDAQ", 3300.00, 3305.00, 3302.50);
        createEquity("TSLA", "US88160R1014", "NASDAQ", 250.00, 251.00, 250.50);
        createEquity("META", "US30303M1027", "NASDAQ", 310.00, 310.50, 310.25);
        createEquity("NVDA", "US67066G1040", "NASDAQ", 450.00, 451.00, 450.50);
        createEquity("JPM", "US46625H1005", "NYSE", 145.00, 145.20, 145.10);
        createEquity("NFLX", "US64110L1061", "NASDAQ", 400.00, 400.10, 400.05);
        createEquity("V", "US92826C8394", "NYSE", 240.00, 240.50, 240.25);
        createEquity("MA", "US57636Q1040", "NYSE", 390.00, 391.00, 390.50);

        // Lending Pool
        lendingService.addToPool(securities.get("MSFT"), 5000); 
        lendingService.setBorrowRate(securities.get("MSFT"), 0.005); 
        
        lendingService.addToPool(securities.get("TSLA"), 2000);
        lendingService.setBorrowRate(securities.get("TSLA"), 0.025);

        lendingService.addToPool(securities.get("NVDA"), 100); 
        lendingService.setBorrowRate(securities.get("NVDA"), 0.05);
        
        lendingService.addToPool(securities.get("NFLX"), 500);
        lendingService.setBorrowRate(securities.get("NFLX"), 0.015);
    }

    @Test
    public void testEquityBuys() {
        placeOrderCheck("AAPL", Side.BUY, 100, 0, "FILLED");
        placeOrderCheck("GOOGL", Side.BUY, 50, 2800.00, "FILLED");
        placeOrderCheck("JPM", Side.BUY, 200, 0, "FILLED");
        placeOrderCheck("NFLX", Side.BUY, 10, 0, "FILLED");
    }

    @Test
    public void testEquityBuySellLoop() {
        placeOrderCheck("META", Side.BUY, 100, 0, "FILLED");
        placeOrderCheck("META", Side.SELL, 50, 310.50, "FILLED");
    }

    @Test
    public void testEquityShortSelling() {
        // Success cases
        tryShortSell("MSFT", 60, 305.00, true);
        tryShortSell("TSLA", 10, 250.00, true);
        tryShortSell("NFLX", 5, 410.00, true);
        
        // Failure case (insufficient pool)
        tryShortSell("NVDA", 150, 0, false);
    }
}
