package com.trading.dummy.model;

import java.time.LocalDate;

public class Bond extends Security {
    private double faceValue;
    private double couponRate;
    private LocalDate maturityDate;

    public Bond(String symbol, String isin, String currency, double faceValue, double couponRate, LocalDate maturityDate) {
        super(symbol, isin, currency, SecurityType.BOND);
        this.faceValue = faceValue;
        this.couponRate = couponRate;
        this.maturityDate = maturityDate;
    }

    public double getFaceValue() { return faceValue; }
    public double getCouponRate() { return couponRate; }
    public LocalDate getMaturityDate() { return maturityDate; }
}
