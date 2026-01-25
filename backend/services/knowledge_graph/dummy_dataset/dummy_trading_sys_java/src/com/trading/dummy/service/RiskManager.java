package com.trading.dummy.service;

import com.trading.dummy.model.Portfolio;
import com.trading.dummy.model.RiskLimit;
import com.trading.dummy.model.Position;
import java.util.logging.Logger;

public class RiskManager {
    private static final Logger LOGGER = Logger.getLogger(RiskManager.class.getName());
    private RiskLimit limit;

    public RiskManager(RiskLimit limit) {
        this.limit = limit;
    }

    public boolean checkRisk(Portfolio portfolio, Pricer pricer) {
        double grossExposure = 0;
        double netExposure = 0;

        for (Position pos : portfolio.getAllPositions()) {
            double value = pricer.getValue(pos.getSecurity(), pos.getQuantity());
            grossExposure += Math.abs(value);
            netExposure += value;
        }

        if (grossExposure > limit.getMaxGrossExposure()) {
            LOGGER.warning("Risk Check Failed: Gross Exposure " + grossExposure + " > " + limit.getMaxGrossExposure());
            return false;
        }
        
        // Similarly check net exposure
        return true;
    }
}
