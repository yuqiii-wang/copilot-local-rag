package com.trading.dummy.model;

import java.time.LocalDate;

public class Option extends Security {
    private Security underlying;
    private double strikePrice;
    private LocalDate expirationDate;
    private OptionType optionType; // Create this enum

    public Option(String symbol, String isin, String currency, Security underlying, double strikePrice, LocalDate expirationDate, OptionType optionType) {
        super(symbol, isin, currency, SecurityType.OPTION);
        this.underlying = underlying;
        this.strikePrice = strikePrice;
        this.expirationDate = expirationDate;
        this.optionType = optionType;
    }

    public Security getUnderlying() { return underlying; }
    public double getStrikePrice() { return strikePrice; }
    public LocalDate getExpirationDate() { return expirationDate; }
    public OptionType getOptionType() { return optionType; }
}
