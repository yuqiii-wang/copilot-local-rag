package com.trading.dummy;

import com.trading.dummy.model.Side;
import org.junit.Test;
import java.time.LocalDate;

public class BondTest extends DummyTestBase {

    @Override
    public void setUp() {
        super.setUp();
        // Load Bonds
        createBond("UST10Y", "US1234567890", 1000, 0.02, LocalDate.now().plusYears(10), 980.00, 980.50, 980.25);
        createBond("UST2Y", "US1234567891", 1000, 0.015, LocalDate.now().plusYears(2), 995.00, 995.20, 995.10);
        createBond("AAPL25", "US037833AG00", 1000, 0.035, LocalDate.now().plusYears(5), 1010.00, 1015.00, 1012.50);
        createBond("JPM30", "US46625HABCD", 1000, 0.045, LocalDate.now().plusYears(30), 1050.00, 1055.00, 1052.50);
        createBond("TSLA28", "US88160RXY12", 1000, 0.030, LocalDate.now().plusYears(8), 950.00, 955.00, 952.50);
    }

    @Test
    public void testBondTrading() {
        placeOrderCheck("UST10Y", Side.BUY, 10, 0, "FILLED");
        placeOrderCheck("AAPL25", Side.BUY, 20, 0, "FILLED");
        placeOrderCheck("JPM30", Side.BUY, 5, 0, "FILLED");
    }
}
