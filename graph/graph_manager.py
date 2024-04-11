from .social_graph import SocialGraph
from datastruct.relation_data import RelationType, RelationChangeType, RelationChange

class GraphManager:
    def __init__(self):
        self.social_graph = SocialGraph()

    def recive(self):
        ...

    def send(self):
        ...

    def update(self):
        ...