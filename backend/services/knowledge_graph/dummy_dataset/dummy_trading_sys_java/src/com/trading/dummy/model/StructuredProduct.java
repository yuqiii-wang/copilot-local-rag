package com.trading.dummy.model;

public class StructuredProduct extends Security {
    private String description;
    // Simplified structure, could have baskets, barriers etc.
    
    public StructuredProduct(String symbol, String isin, String currency, String description) {
        super(symbol, isin, currency, SecurityType.STRUCTURED_PRODUCT);
        this.description = description;
    }

    public String getDescription() { return description; }
}
