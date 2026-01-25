package com.trading.dummy.core;

import com.trading.dummy.model.*;
import com.trading.dummy.service.*;
import java.util.UUID;
import java.util.logging.Logger;
import java.util.logging.Level;

public class ExecutionEngine {
    private static final Logger LOGGER = Logger.getLogger(ExecutionEngine.class.getName());
    private SimplePricer pricer;
    private Portfolio portfolio;
    private RiskManager riskManager;
    private MarketGateway marketGateway;

    public ExecutionEngine(SimplePricer pricer, Portfolio portfolio, RiskManager riskManager) {
        this.pricer = pricer;
        this.portfolio = portfolio;
        this.riskManager = riskManager;
        this.marketGateway = new MarketGateway(pricer);
    }

    public String placeOrder(Order order) {
        LOGGER.info("Received order: " + order.getSide() + " " + order.getQuantity() + " " + order.getSecurity().getSymbol());
        
        // 1. Validate Order
        if (order.getQuantity() <= 0) {
            LOGGER.warning("Order quantity invalid: " + order.getQuantity());
            return order.getOrderId();
        }

        // 2. Market Check (if Limit)
        double marketPrice = pricer.getPrice(order.getSecurity());
        
        boolean isMarketOrder = order.getPrice() <= 0;
        
        if (!isMarketOrder) {
            if (order.getSide() == Side.BUY && marketPrice > order.getPrice()) {
                 LOGGER.info("Buy Limit " + order.getPrice() + " below Market " + marketPrice + ". Pending.");
                 order.setStatus("PENDING");
                 return order.getOrderId();
            }
            if ((order.getSide() == Side.SELL || order.getSide() == Side.SELL_SHORT) && marketPrice < order.getPrice()) {
                 LOGGER.info("Sell Limit " + order.getPrice() + " above Market " + marketPrice + ". Pending.");
                 order.setStatus("PENDING");
                 return order.getOrderId();
            }
        }
        
        // 3. Pre-trade Risk Check (Simplified)
        if (!riskManager.checkRisk(portfolio, pricer)) {
             LOGGER.severe("Order rejected by Risk Manager");
             order.setStatus("REJECTED_RISK");
             return order.getOrderId();
        }

        // 4. Send to Market Gateway (Simulating external connectivity)
        try {
            MarketResponse response = marketGateway.submitOrder(order);
            
            // Handle Gateway Response
            if (!response.isSuccess()) {
                LOGGER.warning("Market rejected order: " + response.getMessage());
                order.setStatus(response.getStatus() != null ? response.getStatus() : "REJECTED_MARKET");
                return order.getOrderId();
            }

            // Handle Messy / Corrupt Data from Market
            if (Double.isNaN(response.getExecutedPrice()) || response.getExecutedPrice() <= 0) {
                 LOGGER.severe("Received corrupt execution price from market: " + response.getExecutedPrice());
                 order.setStatus("ERROR_MARKET_DATA");
                 return order.getOrderId();
            }

            // Valid Execution
            Trade trade = new Trade(UUID.randomUUID().toString(), order.getOrderId(), 
                                    order.getSecurity(), order.getSide(), order.getQuantity(), response.getExecutedPrice());
            
            // 5. Post-trade Update
            portfolio.onTrade(trade);
            order.setStatus("FILLED");
            System.out.println("Order Filled @ " + response.getExecutedPrice() + " (ExchID: " + response.getExchangeOrderId() + ")");
            
            return order.getOrderId();

        } catch (MarketException me) {
            // Handle Network / Latency Issues
            LOGGER.severe("Market Connectivity Issue: " + me.getMessage());
            order.setStatus("REJECTED_NETWORK"); // Or PENDING_RETRY?
            return order.getOrderId();
        }
    }
}
