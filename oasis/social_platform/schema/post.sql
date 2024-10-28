-- This is the schema definition for the post table
-- Add Images, location etc.?
CREATE TABLE post (
    post_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    content TEXT,
    created_at DATETIME,
    num_likes INTEGER DEFAULT 0,
    num_dislikes INTEGER DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES user(user_id)
);
