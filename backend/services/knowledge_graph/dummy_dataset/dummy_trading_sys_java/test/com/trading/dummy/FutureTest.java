package com.trading.dummy;

import com.trading.dummy.model.Side;
import org.junit.Test;
import java.time.LocalDate;

public class FutureTest extends DummyTestBase {

    @Override
    public void setUp() {
        super.setUp();
        // Underlying
        createEquity("MSFT", "US5949181045", "NASDAQ", 300.00, 300.20, 300.10);
        createEquity("AAPL", "US0378331005", "NASDAQ", 150.00, 150.10, 150.05);

        // Futures
        createFuture("ES-DEC24", "USFUTURE1", securities.get("MSFT"), LocalDate.now().plusMonths(6), 50.0, 4500.00, 4500.50, 4500.25); 
        createFuture("NQ-DEC24", "USFUTURE2", securities.get("AAPL"), LocalDate.now().plusMonths(6), 20.0, 15200.00, 15205.00, 15202.50);
    }

    @Test
    public void testFutureTrading() {
        placeOrderCheck("ES-DEC24", Side.BUY, 2, 0, "FILLED");
        placeOrderCheck("NQ-DEC24", Side.BUY, 1, 0, "FILLED");
    }
}
