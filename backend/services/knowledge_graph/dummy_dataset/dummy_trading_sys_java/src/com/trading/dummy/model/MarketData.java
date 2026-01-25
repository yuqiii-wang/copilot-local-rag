package com.trading.dummy.model;

import java.time.LocalDateTime;

public class MarketData {
    private Security security;
    private double bid;
    private double ask;
    private double last;
    private LocalDateTime timestamp;

    public MarketData(Security security, double bid, double ask, double last) {
        this.security = security;
        this.bid = bid;
        this.ask = ask;
        this.last = last;
        this.timestamp = LocalDateTime.now();
    }

    public double getMidPrice() {
        return (bid + ask) / 2.0;
    }

    public Security getSecurity() { return security; }
    public double getLast() { return last; }
}
