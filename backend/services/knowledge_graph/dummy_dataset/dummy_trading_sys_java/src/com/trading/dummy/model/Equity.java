package com.trading.dummy.model;

public class Equity extends Security {
    private String exchange;
    private double dividendYield;

    public Equity(String symbol, String isin, String currency, String exchange) {
        super(symbol, isin, currency, SecurityType.EQUITY);
        this.exchange = exchange;
    }

    public String getExchange() { return exchange; }
    public void setExchange(String exchange) { this.exchange = exchange; }
    public double getDividendYield() { return dividendYield; }
    public void setDividendYield(double dividendYield) { this.dividendYield = dividendYield; }
}
