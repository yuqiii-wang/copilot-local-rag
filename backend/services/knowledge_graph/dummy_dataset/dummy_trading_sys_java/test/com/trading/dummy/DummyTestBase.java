package com.trading.dummy;

import com.trading.dummy.core.ExecutionEngine;
import com.trading.dummy.model.*;
import com.trading.dummy.service.LendingService;
import com.trading.dummy.service.SimplePricer;
import org.junit.Before;
import static org.junit.Assert.*;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

public class DummyTestBase {
    protected static final Logger LOGGER = Logger.getLogger(DummyTestBase.class.getName());

    protected SimplePricer pricer;
    protected LendingService lendingService;
    protected ExecutionEngine engine;
    protected Portfolio portfolio;
    protected RiskManager riskManager;
    protected Map<String, Security> securities;

    @Before
    public void setUp() {
        pricer = new SimplePricer();
        lendingService = new LendingService();
        portfolio = new Portfolio("ACC-TEST-001");
        RiskLimit limits = new RiskLimit(10_000_000, 5_000_000); 
        riskManager = new RiskManager(limits);
        engine = new ExecutionEngine(pricer, portfolio, riskManager);
        securities = new HashMap<>();
    }

    protected void createEquity(String symbol, String isin, String exchange, double bid, double ask, double last) {
        Security sec = new Equity(symbol, isin, "USD", exchange);
        securities.put(symbol, sec);
        pricer.updateMarketData(new MarketData(sec, bid, ask, last));
    }

    protected void createBond(String symbol, String isin, double faceValue, double coupon, LocalDate maturity, double bid, double ask, double last) {
        Security sec = new Bond(symbol, isin, "USD", faceValue, coupon, maturity);
        securities.put(symbol, sec);
        pricer.updateMarketData(new MarketData(sec, bid, ask, last));
    }

    protected void createFuture(String symbol, String isin, Security underlying, LocalDate delivery, double multiplier, double bid, double ask, double last) {
        Security sec = new Future(symbol, isin, "USD", (Equity) underlying, delivery, multiplier); 
        securities.put(symbol, sec);
        pricer.updateMarketData(new MarketData(sec, bid, ask, last));
    }
    
    protected void createOption(String symbol, String isin, Security underlying, double strike, LocalDate expiry, OptionType type, double bid, double ask, double last) {
        Security sec = new Option(symbol, isin, "USD", underlying, strike, expiry, type);
        securities.put(symbol, sec);
        pricer.updateMarketData(new MarketData(sec, bid, ask, last));
    }

    protected void placeOrderCheck(String symbol, Side side, double qty, double price, String expectedStatus) {
        Security sec = securities.get(symbol);
        assertNotNull("Security " + symbol + " not found", sec);
        Order order = new Order(sec, side, qty, price);
        engine.placeOrder(order);
        if ("FILLED".equals(expectedStatus)) {
             assertEquals("Order for " + symbol + " should be FILLED", "FILLED", order.getStatus());
        }
    }

    protected void tryShortSell(String symbol, int qty, double price, boolean expectSuccess) {
        Security sec = securities.get(symbol);
        assertNotNull("Security " + symbol + " not found", sec);
        
        boolean canBorrow = lendingService.canBorrow(sec, qty);
        if (expectSuccess) {
            assertTrue("Should be able to borrow " + symbol, canBorrow);
            lendingService.borrow(sec, qty);
            engine.placeOrder(new Order(sec, Side.SELL_SHORT, qty, price));
        } else {
            assertFalse("Should not be able to borrow " + symbol, canBorrow);
        }
    }
}
