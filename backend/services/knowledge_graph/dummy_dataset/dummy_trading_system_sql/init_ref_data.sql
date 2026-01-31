-- Initial Reference Data for Instruments

INSERT INTO instruments (symbol, type, exchange, tick_size, contract_size) VALUES
('MSFT', 'EQUITY', 'NASDAQ', 0.01, 1),
('AAPL', 'EQUITY', 'NASDAQ', 0.01, 1),
('GOOG', 'EQUITY', 'NASDAQ', 0.01, 1),
('IBM', 'EQUITY', 'NYSE', 0.01, 1),
('TSLA', 'EQUITY', 'NASDAQ', 0.01, 1);

-- Futures
INSERT INTO instruments (symbol, type, exchange, tick_size, contract_size) VALUES
('ESZ3', 'FUTURE', 'CME', 0.25, 50), -- E-mini S&P 500
('NQZ3', 'FUTURE', 'CME', 0.25, 20), -- E-mini Nasdaq
('CLZ3', 'FUTURE', 'NYMEX', 0.01, 1000); -- Crude Oil

-- Dummy Risk Limits
CREATE TABLE risk_limits (
    symbol VARCHAR(10) REFERENCES instruments(symbol),
    max_position_size INT,
    max_daily_loss DECIMAL(18, 2)
);

INSERT INTO risk_limits (symbol, max_position_size, max_daily_loss) VALUES
('MSFT', 10000, 50000.00),
('ESZ3', 500, 100000.00);
