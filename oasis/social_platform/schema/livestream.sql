-- TikTok livestream room table
CREATE TABLE IF NOT EXISTS livestream (
    stream_id INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id INTEGER,
    status TEXT DEFAULT 'live',
    start_time DATETIME,
    end_time DATETIME,
    current_viewers INTEGER DEFAULT 0,
    peak_viewers INTEGER DEFAULT 0,
    total_viewers INTEGER DEFAULT 0,
    total_comments INTEGER DEFAULT 0,
    total_likes INTEGER DEFAULT 0,
    total_gifts_value REAL DEFAULT 0.0,
    FOREIGN KEY(host_id) REFERENCES user(user_id)
);
