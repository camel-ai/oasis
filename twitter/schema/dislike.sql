-- This is the schema definition for the like table
CREATE TABLE dislike (
    dislike_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    tweet_id INTEGER,
    created_at DATETIME,
    FOREIGN KEY(user_id) REFERENCES user(user_id),
    FOREIGN KEY(tweet_id) REFERENCES tweet(tweet_id)
);
