# File: test_twitterAccount_Relationships.py
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from SandboxTwitterModule.infra.twitter_db import TwitterDB, TwitterAccount

class test_twitterAccount_Relationships(unittest.TestCase):
    def setUp(self):
        # Create database connection
        engine = create_engine('sqlite:///twitter_simulation.db')
        Session = sessionmaker(bind=engine)
        self.session = Session()

    def tearDown(self):
        # Close database connection
        self.session.close()

    def test_followers_relationship(self):
        try:
            # Get all users
            users = self.session.query(TwitterAccount).all()

            # Iterate over each user
            for user in users:
                following = user.following or []  # Get the list of users the current user is following
                followers = user.followers or []  # Get the list of users following the current user

                # Check if each follower is in the followers list of the user being followed
                for followed_id in following:
                    followed_account = self.session.query(TwitterAccount).filter_by(user_id=followed_id).one_or_none()
                    
                    # If followed_account is None, skip the assertion
                    if followed_account is None:
                        continue

                    # Convert user_id to string before checking if it's in followed_account.followers
                    self.assertIn(str(user.user_id), followed_account.followers or [],
                                  f"User {user.user_id} follows user {followed_id}, but user {followed_id} does not have user {user.user_id} in followers.")
        except Exception as e:
            self.fail(f"An exception occurred: {e}")

if __name__ == '__main__':
    unittest.main()
