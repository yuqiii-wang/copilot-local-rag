package com.trading.dummy.model;

public class MarginRequirement {
    private double initialMargin;
    private double maintenanceMargin;

    public MarginRequirement(double initialMargin, double maintenanceMargin) {
        this.initialMargin = initialMargin;
        this.maintenanceMargin = maintenanceMargin;
    }

    public double getInitialMargin() { return initialMargin; }
    public double getMaintenanceMargin() { return maintenanceMargin; }
}
