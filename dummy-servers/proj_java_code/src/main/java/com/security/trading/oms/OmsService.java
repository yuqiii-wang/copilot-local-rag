package com.security.trading.oms;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class OmsService {

    private static final Logger logger = LogManager.getLogger(OmsService.class);

    // Threshold above which algorithmic execution is preferred over DMA
    private static final int ALGO_EXECUTION_THRESHOLD = 5000;
    // Threshold above which block-trade routing applies
    private static final double BLOCK_TRADE_NOTIONAL = 500_000.0;

    private final ConcurrentHashMap<String, Order> orders = new ConcurrentHashMap<>();

    public Order createOrder(Order order) {
        if (order.getOrderId() == null) {
            order.setOrderId("OMS-" + LocalDateTime.now().getYear()
                    + "-" + String.format("%05d", orders.size() + 1));
        }
        LocalDateTime now = LocalDateTime.now();
        order.setCreatedAt(now);
        order.setUpdatedAt(now);
        if (order.getStatus() == null) {
            order.setStatus("PENDING");
        }

        logger.info("[{}] New order received: symbol={}, side={}, qty={}, price={}, type={}, TIF={}, algo={}",
                order.getOrderId(), order.getSymbol(), order.getSide(),
                order.getQuantity(), order.getPrice(), order.getOrderType(),
                order.getTimeInForce(), order.getAlgorithm());

        // Routing decision
        double notional = order.getQuantity() * order.getPrice();
        String route = determineRoute(order, notional);
        order.setRoutingInstructions(route);
        logger.info("[{}] Routing decision: notional={}, route={}", order.getOrderId(),
                String.format("%.2f", notional), route);

        orders.put(order.getOrderId(), order);
        simulateOrderExecution(order);
        return order;
    }

    private String determineRoute(Order order, double notional) {
        if (notional >= BLOCK_TRADE_NOTIONAL) {
            logger.info("[{}] Block trade detected (notional={} >= {}): routing to DARK_POOL",
                    order.getOrderId(), String.format("%.2f", notional), BLOCK_TRADE_NOTIONAL);
            return "DARK_POOL";
        }
        if (order.getQuantity() >= ALGO_EXECUTION_THRESHOLD || order.getAlgorithm() != null) {
            String algo = order.getAlgorithm() != null ? order.getAlgorithm() : "VWAP";
            logger.info("[{}] Large order (qty={} >= {}) or algo specified: routing to ALGO_ENGINE with {}",
                    order.getOrderId(), order.getQuantity(), ALGO_EXECUTION_THRESHOLD, algo);
            return "ALGO_ENGINE:" + algo;
        }
        if ("MARKET".equalsIgnoreCase(order.getOrderType())) {
            logger.debug("[{}] Market order, routing to DMA", order.getOrderId());
            return "DMA";
        }
        logger.debug("[{}] Limit order, routing to SMART_ORDER_ROUTER", order.getOrderId());
        return "SMART_ORDER_ROUTER";
    }

    public Order cancelOrder(String orderId) {
        Order order = orders.get(orderId);
        if (order != null) {
            if ("EXECUTED".equals(order.getStatus())) {
                logger.warn("[{}] Cancel request on already EXECUTED order — cancellation rejected", orderId);
            } else {
                order.setStatus("CANCELLED");
                order.setUpdatedAt(LocalDateTime.now());
                logger.info("[{}] Order cancelled", orderId);
            }
        } else {
            logger.error("[{}] Cancel request for unknown orderId", orderId);
        }
        return order;
    }

    private void simulateOrderExecution(Order order) {
        new Thread(() -> {
            try {
                int delayMs = 200 + (int) (Math.random() * 600);
                Thread.sleep(delayMs);
                double slippage = computeSlippage(order);
                double fillPrice = order.getPrice() * (1.0 + slippage);
                order.setStatus("EXECUTED");
                order.setFilledQuantity(order.getQuantity());
                order.setAveragePrice(fillPrice);
                order.setUpdatedAt(LocalDateTime.now());
                logger.info("[{}] Order EXECUTED: fillPrice={} (slippage={}bps), latencyMs={}",
                        order.getOrderId(), String.format("%.4f", fillPrice),
                        String.format("%.2f", slippage * 10_000), delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.error("[{}] Execution thread interrupted", order.getOrderId());
            }
        }).start();
    }

    // Simplified slippage: proportional to sqrt(qty/ADV)
    private double computeSlippage(Order order) {
        double adv = 50_000.0;
        double slippage = 0.0002 * Math.sqrt((double) order.getQuantity() / adv);
        logger.debug("[{}] Slippage estimate: 0.02% * sqrt({}/{}) = {}bps",
                order.getOrderId(), order.getQuantity(), adv,
                String.format("%.4f", slippage * 10_000));
        return slippage;
    }

    public List<Order> getOrders() {
        return new ArrayList<>(orders.values());
    }

    public Order getOrder(String orderId) {
        return orders.get(orderId);
    }

    public List<Order> getOrdersByTradeId(String tradeId) {
        List<Order> result = new ArrayList<>();
        for (Order order : orders.values()) {
            if (tradeId.equals(order.getTradeId())) {
                result.add(order);
            }
        }
        return result;
    }
}
