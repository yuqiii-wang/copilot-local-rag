package com.trading.dummy.service;

import com.trading.dummy.model.Collateral;
import java.util.ArrayList;
import java.util.List;

public class CollateralManager {
    private List<Collateral> postedCollateral = new ArrayList<>();

    public void postCollateral(Collateral collateral) {
        postedCollateral.add(collateral);
    }

    public double getTotalCollateralValue() {
        return postedCollateral.stream().mapToDouble(Collateral::getValuation).sum();
    }
    
    public boolean checkMarginCall(double maintenanceMargin) {
        return getTotalCollateralValue() < maintenanceMargin;
    }
}
