import analysis_prob
import analysis_score

if __name__ == "__main__":
    folder_path = "experiments/reddit/results_analysis/result_data"
    exp_name = "all_360_follow"
    db_path = folder_path + f"/{exp_name}.db"
    exp_info_file_path = folder_path + f"/{exp_name}.json"
    analysis_prob.main(exp_info_file_path, db_path, exp_name, folder_path)
    analysis_score.main(exp_info_file_path, db_path, exp_name, folder_path)
    # analysis_user.main(exp_info_file_path, db_path, exp_name, folder_path)
