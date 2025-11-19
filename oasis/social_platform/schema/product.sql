-- This is the schema definition for the product table
CREATE TABLE IF NOT EXISTS product (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,
    sales INTEGER DEFAULT 0
);
