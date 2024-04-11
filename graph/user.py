from datastruct.agent_data import PersonalInfo, Relationship 

class User:
    def __init__(self, personal_info, relationship):
        self.info = personal_info
        self.relationship = relationship
