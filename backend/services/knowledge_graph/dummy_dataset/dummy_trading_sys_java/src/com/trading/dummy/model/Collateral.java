package com.trading.dummy.model;

public class Collateral {
    private CollateralType type;
    private double amount;
    private String currency; // For CASH
    private double haircut; // % discount

    public Collateral(CollateralType type, double amount, String currency, double haircut) {
        this.type = type;
        this.amount = amount;
        this.currency = currency;
        this.haircut = haircut;
    }

    public double getValuation() {
        return amount * (1 - haircut);
    }
}
