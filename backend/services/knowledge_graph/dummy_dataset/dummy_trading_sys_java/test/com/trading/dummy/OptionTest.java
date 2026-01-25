package com.trading.dummy;

import com.trading.dummy.model.OptionType;
import com.trading.dummy.model.Side;
import org.junit.Test;
import java.time.LocalDate;

public class OptionTest extends DummyTestBase {

    @Override
    public void setUp() {
        super.setUp();
        // Underlying Equities need to exist for Options usually, though our simple model might just link objects
        createEquity("AAPL", "US0378331005", "NASDAQ", 150.00, 150.10, 150.05);
        createEquity("TSLA", "US88160R1014", "NASDAQ", 250.00, 251.00, 250.50);
        createEquity("NFLX", "US64110L1061", "NASDAQ", 400.00, 400.10, 400.05);
        createEquity("DIS", "US2546871060", "NYSE", 90.00, 90.50, 90.25);

        // Options
        createOption("AAPL-OCT-160-C", "USOPTIONS1", securities.get("AAPL"), 160.0, LocalDate.now().plusMonths(3), OptionType.CALL, 5.00, 5.20, 5.10);
        createOption("TSLA-NOV-200-P", "USOPTIONS2", securities.get("TSLA"), 200.0, LocalDate.now().plusMonths(4), OptionType.PUT, 12.00, 12.50, 12.25);
        createOption("NFLX-DEC-420-C", "USOPTIONS3", securities.get("NFLX"), 420.0, LocalDate.now().plusMonths(5), OptionType.CALL, 15.00, 15.50, 15.25);
        createOption("DIS-JAN-85-P", "USOPTIONS4", securities.get("DIS"), 85.0, LocalDate.now().plusMonths(6), OptionType.PUT, 2.50, 2.75, 2.60);
    }

    @Test
    public void testOptionTrading() {
        placeOrderCheck("AAPL-OCT-160-C", Side.BUY, 5, 0, "FILLED");
        placeOrderCheck("TSLA-NOV-200-P", Side.BUY, 10, 0, "FILLED");
        placeOrderCheck("DIS-JAN-85-P", Side.BUY, 50, 0, "FILLED");
        placeOrderCheck("NFLX-DEC-420-C", Side.BUY, 2, 0, "FILLED");
    }
}
