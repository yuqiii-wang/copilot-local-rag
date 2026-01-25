package com.trading.dummy.core;

import com.trading.dummy.model.Order;
import com.trading.dummy.service.SimplePricer;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Logger;

public class MarketGateway {
    private static final Logger LOGGER = Logger.getLogger(MarketGateway.class.getName());
    private Random random = new Random();
    private SimplePricer pricer; // To fetch realistic market prices

    public MarketGateway(SimplePricer pricer) {
        this.pricer = pricer;
    }

    public MarketResponse submitOrder(Order order) throws MarketException {
        // 1. Random Network Latency
        int latency = random.nextInt(2500); // 0 to 2500ms
        try {
            LOGGER.info("Sending order to exchange...");
            Thread.sleep(latency);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new MarketException("Result processing interrupted", e);
        }

        // 2. High Latency Check at Network Layer
        if (latency > 2000) {
            LOGGER.severe("Network timeout waiting for exchange response. Latency: " + latency + "ms");
            throw new MarketException("Network Timeout - No ACK from Exchange");
        }

        // 3. Simulated Network Failures (Packet Loss)
        if (random.nextDouble() < 0.10) {
            throw new MarketException("Connection dropped during transmission");
        }

        // 4. Garbage Data Simulation (Messy Data)
        if (random.nextDouble() < 0.05) {
            // Return nonsense that parsing logic should handle or fail on
            return new MarketResponse(true, "%#Error_Decode_Fail", -1.0, "UNKNOWN", null);
        }

        // 5. Exchange Logic Simulation
        double marketPrice = pricer.getPrice(order.getSecurity());
        
        // Simulating "Market Closed" or "Trading Halt"
        if (random.nextDouble() < 0.05) {
             return new MarketResponse(false, "Exchange Closed / Halted", 0.0, "REJECTED_EXCHANGE", null);
        }

        // Execution Logic
        double execPrice = order.getPrice() > 0 ? order.getPrice() : marketPrice;
        // Maybe we get a better price?
        if (random.nextDouble() < 0.5) {
             execPrice = marketPrice;
        }

        // Messy Data: corrupt price sent back
        if (random.nextDouble() < 0.02) {
             execPrice = Double.NaN;
        }

        return new MarketResponse(
            true, 
            "Order Accepted", 
            execPrice, 
            "FILLED", 
            UUID.randomUUID().toString()
        );
    }
}
