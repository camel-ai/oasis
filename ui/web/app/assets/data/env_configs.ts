export interface typeEnvSettings {
  agent_count: number;
  social_media_style: 'reddit' | 'twitter';
  actions_scope: [];
}
export interface typeEnvAgentConfig {
  title: string;
  // config items name
  controls: (string | undefined)[];
  // config item value
  defaults: (number | undefined)[];
  // nanoid for config items, length equals to controls/defaults
  keys?: string[];
  // default config as mbti, gender, age, profession, country
  lock?: boolean;
  // when temporary config panel saved
  saved?: boolean;
  // when config panel checked to send to python
  checked?: boolean;
}

export const ENV_AGENT_PARAMS = [
  {
    title: 'Gender',
    controls: ['female', 'male'],
    defaults: [0.351, 0.636],
    lock: true,
  },
  {
    title: 'Age',
    controls: ['18-29', '30-49', '50-64', '65-100', 'underage'],
    defaults: [0.44, 0.31, 0.11, 0.03, 0.11],
    lock: true,
  },
  {
    title: 'MBTI',
    controls: [
      'ISTJ',
      'ISFJ',
      'INFJ',
      'INTJ',
      'ISTP',
      'ISFP',
      'INFP',
      'INTP',
      'ESTP',
      'ESFP',
      'ENFP',
      'ENTP',
      'ESTJ',
      'ESFJ',
      'ENFJ',
      'ENTJ',
    ],
    defaults: [
      0.12625, 0.11625, 0.02125, 0.03125, 0.05125, 0.07125, 0.04625, 0.04125,
      0.04625, 0.06625, 0.07125, 0.03625, 0.10125, 0.11125, 0.03125, 0.03125,
    ],
    lock: true,
  },
  {
    title: 'Profession',
    controls: [
      'Agriculture, Food & Natural Resources',
      'Architecture & Construction',
      'Arts, Audio/Video Technology & Communications',
      'Business Management & Administration',
      'Education & Training',
      'Finance',
      'Government & Public Administration',
      'Health Science',
      'Hospitality & Tourism',
      'Human Services',
      'Information Technology',
      'Law, Public Safety, Corrections & Security',
      'Manufacturing',
      'Marketing',
      'Science, Technology, Engineering & Mathematics',
      'Transportation, Distribution & Logistics',
    ],
    defaults: [
      0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
      0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
    ],
    lock: true,
  },
];

export const ENV_AGENT_INTERESTS = [
  {
    name: 'Economics',
    desc: 'The study and management of production, distribution, and consumption of goods and services. Economics focuses on how individuals, businesses, governments, and nations make choices about allocating resources to satisfy their wants and needs, and tries to determine how these groups should organize and coordinate efforts to achieve maximum output.',
    lock: true,
  },
  {
    name: 'IT (Information Technology)',
    desc: 'The use of computers, networking, and other physical devices, infrastructure, and processes to create, process, store, secure, and exchange all forms of electronic data. IT is commonly used within the context of business operations as opposed to personal or entertainment technologies.',
    lock: true,
  },
  {
    name: 'Culture & Society',
    desc: 'The way of life for an entire society, including codes of manners, dress, language, religion, rituals, norms of behavior, and systems of belief. This topic explores how cultural expressions and societal structures influence human behavior, relationships, and social norms.',
    lock: true,
  },
  {
    name: 'General News',
    desc: 'A broad category that includes current events, happenings, and trends across a wide range of areas such as politics, business, science, technology, and entertainment. General news provides a comprehensive overview of the latest developments affecting the world at large.',
    lock: true,
  },
  {
    name: 'Politics',
    desc: 'The activities associated with the governance of a country or other area, especially the debate or conflict among individuals or parties having or hoping to achieve power. Politics is often a battle over control of resources, policy decisions, and the direction of societal norms.',
    lock: true,
  },
  {
    name: 'Business',
    desc: `The practice of making one's living through commerce, trade, or services. This topic encompasses the entrepreneurial, managerial, and administrative processes involved in starting, managing, and growing a business entity.`,
    lock: true,
  },
  {
    name: 'Fun',
    desc: `Activities or ideas that are light-hearted or amusing. This topic covers a wide range of entertainment choices and leisure activities that bring joy, laughter, and enjoyment to individuals and groups.`,
    lock: true,
  },
];

export const ENV_AGENT_PROMPTS = `Please generate a social media user profile based on the provided personal information, including a real name, username, user bio, and a new user persona. The focus should be on creating a fictional background story and detailed interests based on their hobbies and profession.
Output:
{{
    "realname": "str",
    "username": "str",
    "bio": "str",
    "persona": "str"
}}`;
