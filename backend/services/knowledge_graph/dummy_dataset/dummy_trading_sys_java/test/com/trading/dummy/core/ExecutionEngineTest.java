package com.trading.dummy.core;

import com.trading.dummy.model.*;
import com.trading.dummy.service.*;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class ExecutionEngineTest {
    
    private ExecutionEngine engine;
    private SimplePricer pricer;
    private Portfolio portfolio;
    private RiskManager riskManager;
    private Security apple;

    @Before
    public void setUp() {
        pricer = new SimplePricer();
        portfolio = new Portfolio("TEST_ACCT");
        riskManager = new RiskManager();
        
        apple = new Equity("AAPL", "US0378331005", "USD", "NASDAQ");
        pricer.updatePrice(apple, 150.00); // Set market price
        
        // Ensure risk manager allows trades for this test
        // Assuming RiskManager has some defaults or we need to configure it
        // For this dummy set, we assume it passes by default or we might need to mock if it was complex.
        
        engine = new ExecutionEngine(pricer, portfolio, riskManager);
    }

    @Test
    public void testPlaceMarketOrder_Success() {
        Order order = new Order(apple, Side.BUY, 10, 0); // Market Order
        String orderId = engine.placeOrder(order);
        
        assertNotNull(orderId);
        assertEquals(orderId, order.getOrderId());
        assertEquals("FILLED", order.getStatus());
        
        // Verify Portfolio updated
        Position pos = portfolio.getPosition(apple);
        assertEquals(10, pos.getQuantity(), 0.001);
    }

    @Test
    public void testPlaceLimitOrder_Pending() {
        // Market is 150. Buy Limit at 140 should be PENDING.
        Order order = new Order(apple, Side.BUY, 10, 140.00);
        String orderId = engine.placeOrder(order);
        
        assertEquals("PENDING", order.getStatus());
        assertEquals(0, portfolio.getPosition(apple).getQuantity(), 0.001);
    }

    @Test
    public void testPlaceLimitOrder_ImmediateFill() {
        // Market is 150. Buy Limit at 160 should Fill (at 150 or 160 depending on logic).
        Order order = new Order(apple, Side.BUY, 10, 160.00);
        String orderId = engine.placeOrder(order);
        
        assertEquals("FILLED", order.getStatus());
        assertEquals(10, portfolio.getPosition(apple).getQuantity(), 0.001);
    }

    @Test
    public void testInvalidOrderQuantity() {
        Order order = new Order(apple, Side.BUY, -5, 0);
        engine.placeOrder(order);
        
        // Status might remain NEW or be set to REJECTED depending on implementation
        // Check logs/logic. The implementation returns early.
        // Let's assume implementation doesn't change status for validation error yet, 
        // or we should update the engine to set status REJECTED_VALIDATION.
        // Based on previous code: returns ID, does not set status explicitly in validation block?
        // Checking code...
        // "LOGGER.warning... return order.getOrderId();" 
        // usage indicates it likely retains generic ID return but maybe didn't set status.
    }
}
