-- TikTok e-commerce order table
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    buyer_id INTEGER,
    product_id INTEGER,
    source_type TEXT,
    source_id INTEGER,
    price REAL,
    status TEXT DEFAULT 'paid',
    created_at DATETIME,
    FOREIGN KEY(buyer_id) REFERENCES user(user_id),
    FOREIGN KEY(product_id) REFERENCES product(product_id)
);
