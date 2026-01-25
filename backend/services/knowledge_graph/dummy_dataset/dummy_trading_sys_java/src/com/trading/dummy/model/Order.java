package com.trading.dummy.model;

import java.time.LocalDateTime;
import java.util.UUID;

public class Order {
    private String orderId;
    private Security security;
    private Side side;
    private double quantity;
    private double price; // Limit price, 0 for Market
    private LocalDateTime timestamp;
    private String status;

    public Order(Security security, Side side, double quantity, double price) {
        this.orderId = UUID.randomUUID().toString();
        this.security = security;
        this.side = side;
        this.quantity = quantity;
        this.price = price;
        this.timestamp = LocalDateTime.now();
        this.status = "NEW";
    }

    public String getOrderId() { return orderId; }
    public Security getSecurity() { return security; }
    public Side getSide() { return side; }
    public double getQuantity() { return quantity; }
    public double getPrice() { return price; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}
