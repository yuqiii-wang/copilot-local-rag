-- Daily PnL Report Procedure

CREATE OR REPLACE VIEW daily_pnl_view AS
SELECT 
    t.account_id,
    a.username,
    i.type as asset_class,
    SUM(CASE WHEN t.side = 'SELL' THEN t.price * t.quantity ELSE -t.price * t.quantity END) as realized_pnl,
    SUM(t.quantity) as volume_traded
FROM executions e
JOIN orders o ON e.order_id = o.order_id
JOIN accounts a ON o.account_id = a.account_id
JOIN instruments i ON o.symbol = i.symbol
WHERE DATE(e.exec_time) = CURRENT_DATE
GROUP BY t.account_id, a.username, i.type;

-- Query Highest Loss
SELECT * FROM daily_pnl_view ORDER BY realized_pnl ASC LIMIT 5;

-- Volume weighted average price (VWAP) check
SELECT 
    symbol, 
    SUM(price * quantity) / SUM(quantity) as vwap 
FROM executions 
WHERE exec_time > NOW() - INTERVAL '1 hour'
GROUP BY symbol;
