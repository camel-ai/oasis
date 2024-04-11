from enum import Enum

class RelationType(Enum):
    FRIEND = 1
    FAMILY = 2
    COLLEAGUE = 3
    STRANGER = 4

class RelationChangeType(Enum):
    ADD = 1
    REMOVE = 2
    UPDATE = 3

class RelationChange:
    user_id: int
    target_id: int
    change_type: RelationChangeType
    