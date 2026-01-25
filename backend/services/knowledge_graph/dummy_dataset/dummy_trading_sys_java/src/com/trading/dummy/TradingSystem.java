package com.trading.dummy;

import com.trading.dummy.model.*;
import com.trading.dummy.service.*;
import com.trading.dummy.core.ExecutionEngine;
import java.util.Map;
import java.util.logging.Logger;

public class TradingSystem {
    private static final Logger LOGGER = Logger.getLogger(TradingSystem.class.getName());

    public static void main(String[] args) {
        SimplePricer pricer = new SimplePricer();
        
        // Lending Service Setup
        LendingService lendingService = new LendingService();

        // Dummy Data has been migrated to unit tests.
        // See: test/com/trading/dummy/EquityTest.java, BondTest.java, etc.
        LOGGER.info("System initialized (Dummy Data is now in Test Suite).");

        // Check Portfolio
        LOGGER.info("\n--- Portfolio Positions ---");
        for(Position pos : portfolio.getAllPositions()) {
            LOGGER.info(pos.getSecurity().getSymbol() + ": " + pos.getQuantity() + " @ " + pos.getAverageCost() + " | PnL: " + pos.getRealizedPnL());
        }
        
        // Margin
        MarginService marginService = new MarginService();
        MarginRequirement req = marginService.calculateMargin(portfolio, pricer);
        LOGGER.info("\n--- Margin Requirements ---");
        LOGGER.info("Initial Margin: " + req.getInitialMargin());
        LOGGER.info("Maintenance Margin: " + req.getMaintenanceMargin());
        
        // Borrow Cost Example (MSFT)
        if (securities.containsKey("MSFT")) {
           Security msft = securities.get("MSFT");
           LOGGER.info("\n--- Estimated Daily Borrow Cost for MSFT position ---");
           // Assuming we are short 60 (from dummy trades)
           double simpleBorrowCost = lendingService.calculateBorrowCost(msft, 60, 1);
           LOGGER.info("Cost: " + simpleBorrowCost + " (using simplified quantity-based rate)");
        }
    }
}
