package com.trading.dummy.model;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Collection;

public class Portfolio {
    private String accountId;
    private Map<String, Position> positions;

    public Portfolio(String accountId) {
        this.accountId = accountId;
        this.positions = new ConcurrentHashMap<>();
    }

    public Position getPosition(Security security) {
        return positions.computeIfAbsent(security.getSymbol(), k -> new Position(security));
    }
    
    public Collection<Position> getAllPositions() {
        return positions.values();
    }
    
    public void onTrade(Trade trade) {
        Position position = getPosition(trade.getSecurity());
        position.update(trade);
    }
}
