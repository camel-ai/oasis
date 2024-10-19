import analysis_score
import analysis_prob
import analysis_user


if __name__ == "__main__":
    folder_path = (
        '/home/yangz0h/CAMEL-SS/test-main-edition/social-simulation/experiments/reddit/results_analysis/results_analysis/result_data'
    )
    exp_name = 'all_360_follow'
    db_path = folder_path + f"/{exp_name}.db"
    exp_info_file_path = folder_path + f"/{exp_name}.json"
    analysis_prob.main(exp_info_file_path, db_path, exp_name, folder_path)
    analysis_score.main(exp_info_file_path, db_path, exp_name, folder_path)
    # analysis_user.main(exp_info_file_path, db_path, exp_name, folder_path)
