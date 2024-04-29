# to-do list

- [ ] 完成 recsys 模块的初步实现
    - [ ] def fetch_data(db_file): 从sqlite数据库读取 table user 和 table tweets     
    - [x] def clean_data(tweets): 对获取的数据进行必要的清洗和预处理，包括过滤掉违法内容、标准化文本格式等。
    - [x] def calculate_similarity(tweets): 相似度计算
    - [ ] def store_recommendations(db_file, user_recommendations): 根据计算得出的相似度或推荐评分，为每个用户生成推荐列表，并将这些推荐写入到 rec 表中。
- [ ] 形成 recsys 部分的代码规范
    - [ ] 需要一个新的 feature 用于创造一些数据进行 unit test
    - [ ] 使用 GitHub action 进行 unit test, CI/CD, etc
    

## quick-start

- python -m venv venv_name (like simula)
- source simula/bin/activate (for Mac OS X)
- pip install --upgrade pip  
- pip install -r code/env/requirements.txt

## unit test

- [ ] still waiting for the new feature(data generator)

## tips for commits

Please replace 'venv' with the name of your own virtual environment in the .gitignore file located at the root of the SOCIAL-SIMULATION
