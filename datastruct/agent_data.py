class PersonalInfo:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f'{self.name} is {self.age} years old.'
    
    def __repr__(self):
        return f'PersonalInfo({self.name}, {self.age})'
    
    def update(self):
        ...

class Relationship:
    ...

