package com.trading.dummy.model;

import java.time.LocalDateTime;

public class Trade {
    private String tradeId;
    private String orderId;
    private Security security;
    private Side side;
    private double quantity;
    private double price;
    private LocalDateTime tradeDate;

    public Trade(String tradeId, String orderId, Security security, Side side, double quantity, double price) {
        this.tradeId = tradeId;
        this.orderId = orderId;
        this.security = security;
        this.side = side;
        this.quantity = quantity;
        this.price = price;
        this.tradeDate = LocalDateTime.now();
    }
    
    public String getTradeId() { return tradeId; }
    public String getOrderId() { return orderId; }
    public Security getSecurity() { return security; }
    public Side getSide() { return side; }
    public double getQuantity() { return quantity; }
    public double getPrice() { return price; }
    public LocalDateTime getTradeDate() { return tradeDate; }
}
