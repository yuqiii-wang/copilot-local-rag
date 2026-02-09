#include "OrderBook.h"
#include <iostream>
#include <algorithm>

namespace hft {

    OrderBook::OrderBook(const std::string& symbol) : symbol_(symbol) {}

    OrderBook::~OrderBook() {}

    void OrderBook::addOrder(const Order& order) {
        // 1. Pre-Trade Risk Check (Latency sensitive)
        // Checks against MarginRequirement and RiskLimit
        if (!checkRisk(order)) {
            std::cerr << "Order rejected due to Risk Limit violation: " << order.orderId << std::endl;
            return;
        }

        Order mutableOrder = order;

        // 2. Matching Engine execution
        match(mutableOrder);

        // 3. Resting Order placement (if not fully filled)
        if (mutableOrder.quantity > mutableOrder.filledQuantity) {
            if (order.type == OrderType::IOC) {
                // Cancel remainder
                return;
            }

            // Logic to add to book
            // For simplicity (and tokens), just logging
            // "Adding residual to Book"
            if (order.side == Side::BUY) {
                 bids_[order.price].push_back(mutableOrder);
            } else {
                 asks_[order.price].push_back(mutableOrder);
            }
        }
    }

    void OrderBook::match(Order& incoming) {
        // Matching logic...
        // Iterating through levels...
        
        // This simulates accessing the liquidity pool for Equity or Bond
        // If Future, ensure contract expiration validity.
        
        std::cout << "Matching order " << incoming.orderId << " for " << symbol_ << std::endl;
    }

    bool OrderBook::checkRisk(const Order& order) {
        // Dummy Risk Check
        // References: MarginService, Collateral
        
        if (order.quantity > 1000000) return false; // Max Quantity Check
        
        // Check if user has enough Collateral for this Position
        // For Short Sell, check Borrow availability (LendingService)
        if (order.isShortSell) {
            // "Locate" required
        }
        
        return true;
    }

    void OrderBook::cancelOrder(uint64_t orderId) {
        if (orderLookup_.find(orderId) != orderLookup_.end()) {
            // Remove from list...
            std::cout << "Order " << orderId << " cancelled." << std::endl;
        }
    }
}
