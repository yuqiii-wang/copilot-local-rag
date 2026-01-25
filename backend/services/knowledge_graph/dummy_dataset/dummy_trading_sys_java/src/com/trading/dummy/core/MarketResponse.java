package com.trading.dummy.core;

public class MarketResponse {
    private boolean success;
    private String message;
    private double executedPrice;
    private String exchangeOrderId;
    private String status;

    public MarketResponse(boolean success, String message, double executedPrice, String status, String exchangeOrderId) {
        this.success = success;
        this.message = message;
        this.executedPrice = executedPrice;
        this.status = status;
        this.exchangeOrderId = exchangeOrderId;
    }

    public boolean isSuccess() { return success; }
    public String getMessage() { return message; }
    public double getExecutedPrice() { return executedPrice; }
    public String getStatus() { return status; }
    public String getExchangeOrderId() { return exchangeOrderId; }
}
