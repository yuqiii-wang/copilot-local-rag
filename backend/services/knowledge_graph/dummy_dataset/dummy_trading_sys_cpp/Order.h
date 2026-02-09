#pragma once

#include <string>
#include <cstdint>
#include <chrono>

namespace hft {

    enum class Side {
        BUY,
        SELL
    };

    enum class OrderType {
        LIMIT,
        MARKET,
        IOC, // Immediate or Cancel
        FOK  // Fill or Kill
    };

    // Aligned to cache line for performance
    struct alignas(64) Order {
        uint64_t orderId;
        uint64_t timestamp; // Nanos
        double price;
        uint32_t quantity;
        uint32_t filledQuantity;
        char symbol[8]; // e.g. "MSFT", "GOOG"
        Side side;
        OrderType type;
        uint64_t clientId;
        
        // Biz flags for Risk checks
        bool isMarginTrade;
        bool isShortSell;
    };

    struct ExecutionReport {
        uint64_t orderId;
        uint64_t matchId;
        double price;
        uint32_t quantity;
        uint64_t timestamp;
        char symbol[8];
    };
}
