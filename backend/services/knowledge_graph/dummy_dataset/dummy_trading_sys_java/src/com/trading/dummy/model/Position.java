package com.trading.dummy.model;

public class Position {
    private Security security;
    private double quantity;
    private double averageCost;
    private double realizedPnL;
    
    public Position(Security security) {
        this.security = security;
        this.quantity = 0;
        this.averageCost = 0;
        this.realizedPnL = 0;
    }

    public synchronized void update(Trade trade) {
        if (!trade.getSecurity().equals(this.security)) return;

        double tradeQty = trade.getQuantity();
        double tradePrice = trade.getPrice();
        
        if (trade.getSide() == Side.BUY) {
            double totalCost = (averageCost * quantity) + (tradePrice * tradeQty);
            quantity += tradeQty;
            if (quantity != 0) averageCost = totalCost / quantity;
        } else if (trade.getSide() == Side.SELL) {
             // Simplify: PnL calculation on Sell
             double pnl = (tradePrice - averageCost) * tradeQty;
             realizedPnL += pnl;
             quantity -= tradeQty;
             if (quantity == 0) averageCost = 0;
        }
    }
    
    public Security getSecurity() { return security; }
    public double getQuantity() { return quantity; }
    public double getAverageCost() { return averageCost; }
    public double getRealizedPnL() { return realizedPnL; }
}
