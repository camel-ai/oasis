# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import analysis_score

if __name__ == "__main__":
    folder_path = ("visualization/reddit_simulation_align_with_human"
                   "/experiment_results")
    exp_name = "business_3600"
    db_path = folder_path + f"/{exp_name}.db"
    exp_info_file_path = folder_path + f"/{exp_name}.json"
    analysis_score.main(exp_info_file_path, db_path, exp_name, folder_path)
