#pragma once

#include "Order.h"
#include <map>
#include <list>
#include <vector>
#include <unordered_map>

namespace hft {

    // Forward declaration
    class RiskManager;

    /**
     * OrderBook implementation optimized for low latency.
     * Supports standard Limit Order Book operations.
     * 
     * Specific handling for Equity and Future asset classes.
     */
    class OrderBook {
    public:
        OrderBook(const std::string& symbol);
        ~OrderBook();

        // Core Actions
        void addOrder(const Order& order);
        void cancelOrder(uint64_t orderId);
        void modifyOrder(uint64_t orderId, uint32_t newQuantity);

        // Market Data publication
        void snapshot() const;

    private:
        void match(Order& incomingOrder);
        bool checkRisk(const Order& order);

        std::string symbol_;
        
        // Price Levels: Price -> List of Orders (FIFO)
        // Using map for ordered traversal (Ascending for Ask, Descending logic needed for Bids)
        std::map<double, std::list<Order>> bids_; // High to Low
        std::map<double, std::list<Order>> asks_; // Low to High

        // Fast lookup for modification/cancellation
        std::unordered_map<uint64_t, std::list<Order>::iterator> orderLookup_;
    };
}
