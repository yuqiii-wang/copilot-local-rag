package com.trading.dummy;

import com.trading.dummy.model.Order;
import com.trading.dummy.model.Side;
import org.junit.Test;
import static org.junit.Assert.*;

public class OrderTest extends DummyTestBase {

    @Override
    public void setUp() {
        super.setUp();
        createEquity("AAPL", "US0378331005", "NASDAQ", 150.00, 150.10, 150.05);
    }

    @Test
    public void testMarketOrderExecution() {
        placeOrderCheck("AAPL", Side.BUY, 100, 0, "FILLED");
    }

    @Test
    public void testLimitOrderPlacement() {
        // Market is 150.05 (Last) / 150.10 (Ask). 
        // Buy Limit at 140 should be PENDING.
        Order order = new Order(securities.get("AAPL"), Side.BUY, 10, 140.00);
        engine.placeOrder(order);
        assertEquals("PENDING", order.getStatus());
    }

    @Test
    public void testLimitOrderImmediateFill() {
        // Market is 150.05. Buy Limit at 160 should be FILLED.
        Order order = new Order(securities.get("AAPL"), Side.BUY, 10, 160.00);
        engine.placeOrder(order);
        assertEquals("FILLED", order.getStatus());
    }

    @Test
    public void testOrderSide() {
        Order buy = new Order(securities.get("AAPL"), Side.BUY, 10, 0);
        Order sell = new Order(securities.get("AAPL"), Side.SELL, 10, 0);
        assertEquals(Side.BUY, buy.getSide());
        assertEquals(Side.SELL, sell.getSide());
    }
}
