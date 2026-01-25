package com.trading.dummy.service;

import com.trading.dummy.model.Security;
import com.trading.dummy.model.MarketData;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class SimplePricer implements Pricer {
    private Map<String, MarketData> marketDataMap = new ConcurrentHashMap<>();

    public void updateMarketData(MarketData data) {
        marketDataMap.put(data.getSecurity().getSymbol(), data);
    }

    @Override
    public double getPrice(Security security) {
        MarketData data = marketDataMap.get(security.getSymbol());
        if (data != null) {
            return data.getLast();
        }
        return 0.0;
    }

    @Override
    public double getValue(Security security, double quantity) {
        return getPrice(security) * quantity;
    }
}
