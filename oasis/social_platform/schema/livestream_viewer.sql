-- TikTok livestream viewer session tracking
CREATE TABLE IF NOT EXISTS livestream_viewer (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_id INTEGER,
    viewer_id INTEGER,
    enter_time DATETIME,
    exit_time DATETIME,
    total_stay_seconds REAL DEFAULT 0.0,
    interactions INTEGER DEFAULT 0,
    FOREIGN KEY(stream_id) REFERENCES livestream(stream_id),
    FOREIGN KEY(viewer_id) REFERENCES user(user_id)
);
