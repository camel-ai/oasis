-- This is the schema definition for the rec table
CREATE TABLE rec (
    user_id INTEGER,
    tweet_id INTEGER,
    PRIMARY KEY(user_id, tweet_id),
    FOREIGN KEY(user_id) REFERENCES user(user_id)
    FOREIGN KEY(tweet_id) REFERENCES tweet(tweet_id)
);
