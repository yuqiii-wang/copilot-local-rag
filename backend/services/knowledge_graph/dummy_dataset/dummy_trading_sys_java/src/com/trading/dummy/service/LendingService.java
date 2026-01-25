package com.trading.dummy.service;

import com.trading.dummy.model.Security;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LendingService {
    private Map<Security, Double> lendingPool; // Security -> Quantity available
    private Map<Security, Double> borrowRates; // Security -> Daily Rate

    public LendingService() {
        this.lendingPool = new ConcurrentHashMap<>();
        this.borrowRates = new ConcurrentHashMap<>();
    }

    public void addToPool(Security security, double quantity) {
        lendingPool.merge(security, quantity, Double::sum);
    }

    public void setBorrowRate(Security security, double rate) {
        borrowRates.put(security, rate);
    }

    public boolean canBorrow(Security security, double quantity) {
        return lendingPool.getOrDefault(security, 0.0) >= quantity;
    }

    public void borrow(Security security, double quantity) throws Exception {
        if (!canBorrow(security, quantity)) {
            throw new Exception("Insufficient liquidity to borrow " + security.getSymbol());
        }
        lendingPool.merge(security, -quantity, Double::sum);
    }
    
    public void returnBorrowed(Security security, double quantity) {
        lendingPool.merge(security, quantity, Double::sum);
    }

    public double calculateBorrowCost(Security security, double quantity, int days) {
        double rate = borrowRates.getOrDefault(security, 0.01); // Default 1%
        return quantity * 100 * rate * days / 365.0; // Assuming price ~100 or quantity is value? treating quantity as notion of value for simplicity or rate applies to qty? Let's assume rate is per unit value. But here rate is likely percentage of value. 
        // For simplicity, let's say cost = qty * rate * days. (If rate is cost per unit per day).
        // Standard is Cost = Notional * Rate * Time. We don't have price here. 
        // Let's assume Pricer is needed for accurate cost. 
        // I'll skip Pricer dependency here to keep it decoupled and simple.
        return quantity * rate * days;
    }
}
