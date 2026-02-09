#include <iostream>
#include <vector>
#include "OrderBook.h"
#include "Order.h"

// Strategy: Market Market for high liquidity Equity symbols (e.g. MSFT, AAPL)
// Also handles Futures rolls logic simplified.

int main() {
    std::cout << "Starting HFT Strategy Engine..." << std::endl;

    hft::OrderBook msftBook("MSFT");
    hft::OrderBook esFutureBook("ESZ3"); // E-mini S&P 500 Future

    // Create a Buy Order
    hft::Order buyOrder;
    buyOrder.orderId = 101;
    buyOrder.price = 250.50;
    buyOrder.quantity = 100;
    buyOrder.side = hft::Side::BUY;
    buyOrder.type = hft::OrderType::LIMIT;
    buyOrder.isMarginTrade = true; // Use Margin
    
    msftBook.addOrder(buyOrder);

    // Create a Short Sell Order
    hft::Order shortOrder;
    shortOrder.orderId = 102;
    shortOrder.price = 4500.25;
    shortOrder.quantity = 10;
    shortOrder.side = hft::Side::SELL;
    shortOrder.isShortSell = true;
    shortOrder.type = hft::OrderType::LIMIT;

    esFutureBook.addOrder(shortOrder);

    std::cout << "Strategy Cycle Complete." << std::endl;
    return 0;
}
