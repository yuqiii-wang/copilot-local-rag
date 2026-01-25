package com.trading.dummy.model;

import java.io.Serializable;
import java.util.Objects;

public abstract class Security implements Serializable {
    private String symbol;
    private String isin;
    private String currency;
    private SecurityType type;

    public Security(String symbol, String isin, String currency, SecurityType type) {
        this.symbol = symbol;
        this.isin = isin;
        this.currency = currency;
        this.type = type;
    }

    public String getSymbol() { return symbol; }
    public String getIsin() { return isin; }
    public String getCurrency() { return currency; }
    public SecurityType getType() { return type; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Security security = (Security) o;
        return Objects.equals(symbol, security.symbol) &&
                Objects.equals(isin, security.isin);
    }

    @Override
    public int hashCode() {
        return Objects.hash(symbol, isin);
    }

    @Override
    public String toString() {
        return "Security{" +
                "symbol='" + symbol + '\'' +
                ", isin='" + isin + '\'' +
                ", type=" + type +
                '}';
    }
}
