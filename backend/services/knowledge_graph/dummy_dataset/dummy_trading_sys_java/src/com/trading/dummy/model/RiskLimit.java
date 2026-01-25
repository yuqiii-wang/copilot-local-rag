package com.trading.dummy.model;

public class RiskLimit {
    private double maxGrossExposure;
    private double maxNetExposure;

    public RiskLimit(double maxGrossExposure, double maxNetExposure) {
        this.maxGrossExposure = maxGrossExposure;
        this.maxNetExposure = maxNetExposure;
    }

    public double getMaxGrossExposure() { return maxGrossExposure; }
    public double getMaxNetExposure() { return maxNetExposure; }
}
