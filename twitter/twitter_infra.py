# File: SandboxTwitterModule/infra/twitter_infra.py
import os

from .twitter_db import TwitterDB 


class TwitterInfra:
    def __init__(self):
        self.TwiDB = TwitterDB()  # twitter 自己的 DB

    @property
    def db(self):
        return self.TwiDB
