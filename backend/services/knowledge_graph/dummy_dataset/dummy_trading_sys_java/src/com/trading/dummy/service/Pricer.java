package com.trading.dummy.service;

import com.trading.dummy.model.Security;

public interface Pricer {
    double getPrice(Security security);
    double getValue(Security security, double quantity);
}
