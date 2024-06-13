-- This is the schema definition for the tweet table
-- Add Images, location etc.?
CREATE TABLE tweet (
    tweet_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    content TEXT,
    created_at DATETIME,
    num_likes INTEGER DEFAULT 0,
    num_dislikes INTEGER DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES user(user_id)
);
