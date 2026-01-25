package com.trading.dummy.model;

import java.time.LocalDate;

public class Future extends Security {
    private Security underlying;
    private LocalDate expirationDate;
    private double contractSize;

    public Future(String symbol, String isin, String currency, Security underlying, LocalDate expirationDate, double contractSize) {
        super(symbol, isin, currency, SecurityType.FUTURE);
        this.underlying = underlying;
        this.expirationDate = expirationDate;
        this.contractSize = contractSize;
    }

    public Security getUnderlying() { return underlying; }
    public LocalDate getExpirationDate() { return expirationDate; }
    public double getContractSize() { return contractSize; }
}
