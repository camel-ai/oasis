from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class InterestEnum(str, Enum):
    r"""
    Predefined interest tag enumeration.
    Used to identify users' main areas of interest, helping to categorize and understand user focus points.
    """
    ARTIFICIAL_INTELLIGENCE = "Artificial Intelligence"  # AI technology, machine learning, deep learning, etc.
    SOFTWARE_DEVELOPMENT = "Software Development"  # Software development, programming languages, etc.
    CRYPTO_BLOCKCHAIN = "Cryptocurrency & Blockchain"  # Cryptocurrency, blockchain, etc.
    ECONOMICS_MARKETS = "Economics & Markets"  # Economics, market analysis, macroeconomics, etc.
    INVESTING_FINANCE = "Investing & Personal Finance"  # Investment and financial management, stocks, funds, etc.
    TRAVEL_ADVENTURE = "Travel & Adventure"  # Travel, adventure, etc.
    HEALTH_WELLNESS = "Health & Wellness"  # Health and wellness, medical care, etc.
    GAMING = "Gaming"  # Gaming, etc.
    MOVIES_TV_SHOWS = "Movies & TV Shows"  # Movies, TV series, etc.
    MUSIC = "Music"  # Music, etc.
    BOOKS_LITERATURE = "Books & Literature"  # Books, literature, etc.
    POLITICS_GOVERNANCE = "Politics & Governance"  # Politics, public policy, government administration, etc.
    SOCIAL_ISSUES_ADVOCACY = "Social Issues & Advocacy"  # Social issues, public welfare, etc.
    EDUCATION = "Education"  # Education, learning methods, knowledge sharing, etc.
    PHILOSOPHY = "Philosophy"  # Philosophy, ethics, logic, etc.
    HOSPITALITY_INDUSTRY = "Hospitality Industry"  # Hotels, tourism, etc.
    SUSTAINABILITY = "Sustainability"  # Environmental protection, sustainable development, etc.
    ECO_TOURISM = "Eco-tourism"  # Ecological tourism, etc.
    BUSINESS_STRATEGY = "Business Strategy"  # Business strategy, enterprise management, entrepreneurship, etc.
    TECH_IN_HOSPITALITY = "Technology in Hospitality"  # Hotel technology, tourism technology, etc.
    LOGISTICS = "Logistics and Distribution Networks"  # Logistics, distribution networks, etc.
    CULTURAL_INTERSECTIONS = "Cultural Intersections"  # Cultural intersections, etc.


class InteractionStyleEnum(str, Enum):
    r"""
    Interaction style enumeration.
    Used to describe users' communication style and tone in interactions.
    """
    FRIENDLY_SUPPORTIVE = "Friendly & Supportive"  # Friendly and supportive, providing help and support to others
    FORMAL_PROFESSIONAL = "Formal & Professional"  # Formal and professional, using professional terminology and formal tone
    CASUAL_FRIENDLY = "Casual & Friendly"  # Casual and friendly, relaxed and warm communication style
    ANALYTICAL_INQUISITIVE = "Analytical & Inquisitive"  # Analytical and inquisitive, enjoys deep analysis and asking questions
    SUPPORTIVE_ENCOURAGING = "Supportive & Encouraging"  # Supportive and encouraging, providing positive feedback to others
    HUMOROUS_WITTY = "Humorous & Witty"  # Humorous and witty, good at using humor and wit
    DIRECT_STRAIGHTFORWARD = "Direct & Straightforward"  # Direct and straightforward, speaks frankly
    THOUGHTFUL_REFLECTIVE = "Thoughtful & Reflective"  # Thoughtful and reflective, enjoys reflection and thinking


class ContentStyleEnum(str, Enum):
    r"""
    Content style enumeration.
    Used to describe the style and type of content created by users.
    """
    LONG_FORM_POSTS = "Long-form Posts"  # Long articles, detailed and in-depth content
    SHORT_QUICK_UPDATES = "Short & Quick Updates"  # Brief and quick updates, fragmented information
    VISUAL_CONTENT = "Visual Content"  # Visual content such as images, videos, etc.
    OPINION_EDITORIALS = "Opinion & Editorials"  # Opinion commentary, personal views and positions
    EDUCATIONAL_TUTORIALS = "Educational & Tutorials"  # Educational tutorials, knowledge sharing
    NEWS_UPDATES = "News & Updates"  # News updates, current affairs information
    PERSONAL_STORIES = "Personal Stories"  # Personal stories, life experience sharing
    PROFESSIONAL_INSIGHTS = "Professional Insights"  # Professional insights, industry experience sharing
    QUESTIONS_INQUIRIES = "Questions & Inquiries"  # Questions and inquiries, exploratory content


class ActivePeriodEnum(str, Enum):
    """
    Active period enumeration.
    Used to describe users' main active time periods during the day.
    """
    MORNING = "Morning"  # Morning period, usually 6:00-12:00
    AFTERNOON = "Afternoon"  # Afternoon period, usually 12:00-18:00
    EVENING = "Evening"  # Evening period, usually 18:00-22:00
    NIGHT = "Night"  # Night period, usually 22:00-6:00


class BehavioralArchetypeEnum(str, Enum):
    """
    Behavioral archetype enumeration.
    Used to describe users' main behavioral patterns and role positioning on social platforms.
    """
    CONTENT_CREATOR = "Content Creator"  # Content creator, actively creates and publishes content
    ACTIVE_PARTICIPANT = "Active Participant"  # Active participant, frequently engages in discussions and interactions
    OBSERVER_LURKER = "Observer & Lurker"  # Observer, mainly browses and observes, rarely speaks
    THOUGHT_LEADER = "Thought Leader"  # Thought leader, leads discussions and shares insights
    COMMUNITY_BUILDER = "Community Builder"  # Community builder, dedicated to building and maintaining communities
    KNOWLEDGE_SEEKER = "Knowledge Seeker"  # Knowledge seeker, actively learns and acquires knowledge
    CASUAL_USER = "Casual User"  # Casual user, occasional use, low engagement


class CommunityRoleEnum(str, Enum):
    """
    Community role enumeration.
    Used to describe the roles and status that users may play in a specific community.
    """
    LEADER_ORGANIZER = "Leader & Organizer"  # Leader and organizer, responsible for community management and event organization
    MENTOR_GUIDE = "Mentor & Guide"  # Mentor and guide, provides help and guidance to new members
    EXPERT_CONTRIBUTOR = "Expert Contributor"  # Expert contributor, has professional knowledge and experience in a specific field
    ACTIVE_PARTICIPANT = "Active Participant"  # Active participant, frequently participates in community activities and discussions
    NEWCOMER_LEARNER = "Newcomer & Learner"  # Newcomer and learner, just joined the community or is in the learning stage
