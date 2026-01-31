-- Schema for Trading System

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('TRADER', 'RISK_OFFICER', 'ADMIN')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    currency VARCHAR(3) DEFAULT 'USD',
    balance DECIMAL(18, 2) DEFAULT 0.00,
    margin_balance DECIMAL(18, 2) DEFAULT 0.00
);

CREATE TABLE instruments (
    symbol VARCHAR(10) PRIMARY KEY,
    type VARCHAR(20) CHECK (type IN ('EQUITY', 'FUTURE', 'OPTION', 'BOND')),
    exchange VARCHAR(10),
    tick_size DECIMAL(10, 4),
    contract_size INT DEFAULT 1 -- For futures/options
);

CREATE TABLE orders (
    order_id BIGSERIAL PRIMARY KEY,
    account_id INT REFERENCES accounts(account_id),
    symbol VARCHAR(10) REFERENCES instruments(symbol),
    side VARCHAR(4) CHECK (side IN ('BUY', 'SELL')),
    quantity INT NOT NULL,
    price DECIMAL(18, 4), -- NULL for Market Orders
    type VARCHAR(10) CHECK (type IN ('LIMIT', 'MARKET', 'STOP')),
    status VARCHAR(10) DEFAULT 'NEW',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE executions (
    exec_id BIGSERIAL PRIMARY KEY,
    order_id BIGINT REFERENCES orders(order_id),
    price DECIMAL(18, 4),
    quantity INT,
    exec_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
