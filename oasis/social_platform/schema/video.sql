-- TikTok video metadata extension table (linked to post table)
CREATE TABLE IF NOT EXISTS video (
    post_id INTEGER PRIMARY KEY,
    duration_seconds INTEGER DEFAULT 15,
    category TEXT DEFAULT 'general',
    topic_tags TEXT DEFAULT '[]',
    quality_score REAL DEFAULT 0.5,
    hook_strength REAL DEFAULT 0.5,
    has_product_link BOOLEAN DEFAULT 0,
    product_id INTEGER,
    traffic_pool_level INTEGER DEFAULT 1,
    pool_enter_time DATETIME,
    total_impressions INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    total_watch_ratio REAL DEFAULT 0.0,
    share_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    FOREIGN KEY(post_id) REFERENCES post(post_id),
    FOREIGN KEY(product_id) REFERENCES product(product_id)
);
