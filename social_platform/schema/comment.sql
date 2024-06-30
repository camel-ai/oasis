-- This is the schema definition for the comment table
CREATE TABLE comment (
    comment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tweet_id INTEGER,
    user_id INTEGER,
    content TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    num_likes INTEGER DEFAULT 0,
    num_dislikes INTEGER DEFAULT 0,
    FOREIGN KEY(tweet_id) REFERENCES tweet(tweet_id),
    FOREIGN KEY(user_id) REFERENCES user(user_id)
);
