package com.trading.dummy.core;

import com.trading.dummy.model.Order;
import com.trading.dummy.model.Side;
import com.trading.dummy.model.Security;
import com.trading.dummy.model.Equity;
import com.trading.dummy.service.SimplePricer;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class MarketGatewayTest {

    private MarketGateway gateway;
    private SimplePricer pricer;
    private Security apple;

    @Before
    public void setUp() {
        pricer = new SimplePricer();
        apple = new Equity("AAPL", "US0378331005", "USD", "NASDAQ");
        pricer.updatePrice(apple, 150.00);
        
        gateway = new MarketGateway(pricer);
    }

    @Test
    public void testSubmitOrder_BasicSuccess() {
        Order order = new Order(apple, Side.BUY, 100, 0);
        // Retrying loop because gateway has random failures
        boolean successOnce = false;
        
        for (int i = 0; i < 10; i++) {
            try {
                MarketResponse response = gateway.submitOrder(order);
                if (response.isSuccess()) {
                    assertTrue(response.getExecutedPrice() > 0);
                    assertNotNull(response.getExchangeOrderId());
                    successOnce = true;
                    break;
                }
            } catch (MarketException e) {
                // Expected occasionally
                System.out.println("Ignored expected test failure: " + e.getMessage());
            }
        }
        assertTrue("Should succeed at least once in 10 tries", successOnce);
    }
    
    @Test
    public void testMessyDataSimulation() {
        // This is hard to deterministically test without mocking the Random instance inside MarketGateway.
        // In a real generic test, we might iterate many times to ensure no uncaught exceptions explode.
        Order order = new Order(apple, Side.BUY, 100, 0);
        
        for (int i = 0; i < 50; i++) {
            try {
                gateway.submitOrder(order);
            } catch (MarketException e) {
                // Handled
            } catch (Exception e) {
                fail("Should not throw generic non-handled exceptions: " + e.getClass().getName());
            }
        }
    }
}
