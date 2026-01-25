package com.trading.dummy.service;

import com.trading.dummy.model.Portfolio;
import com.trading.dummy.model.Position;
import com.trading.dummy.model.SecurityType;
import com.trading.dummy.model.MarginRequirement;

public class MarginService {
    
    // Simplistic margin calc
    public MarginRequirement calculateMargin(Portfolio portfolio, Pricer pricer) {
        double totalInitial = 0;
        double totalMaintenance = 0;

        for (Position pos : portfolio.getAllPositions()) {
            double value = pricer.getValue(pos.getSecurity(), pos.getQuantity());
            double absValue = Math.abs(value);
            
            // Dummy logic: 50% IM, 25% MM for Equities
            if (pos.getSecurity().getType() == SecurityType.EQUITY) {
                totalInitial += absValue * 0.50;
                totalMaintenance += absValue * 0.25;
            }
        }
        return new MarginRequirement(totalInitial, totalMaintenance);
    }
}
