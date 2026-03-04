CREATE VIEW vw_daily_orders AS
SELECT
  order_date,
  COUNT(DISTINCT order_id) AS orders
FROM vw_orders_enriched
WHERE order_status IN ('delivered','shipped','invoiced','processing','approved')
GROUP BY order_date
ORDER BY order_date

CREATE VIEW vw_daily_orders_by_category AS
SELECT
  DATE(o.order_purchase_timestamp) AS order_date,
  p.product_category_name AS category,
  COUNT(DISTINCT oi.order_id) AS orders
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE o.order_purchase_timestamp IS NOT NULL
  AND o.order_status IN ('delivered','shipped','invoiced','processing','approved')
GROUP BY DATE(o.order_purchase_timestamp), p.product_category_name
ORDER BY order_date, category

CREATE VIEW vw_daily_orders_revenue AS
SELECT
  e.order_date,
  COUNT(DISTINCT e.order_id) AS orders,
  COALESCE(SUM(t.gross_value), 0) AS revenue
FROM vw_orders_enriched e
LEFT JOIN vw_order_totals t ON t.order_id = e.order_id
WHERE e.order_status IN ('delivered','shipped','invoiced','processing','approved')
GROUP BY e.order_date
ORDER BY e.order_date
