import analysis_score

if __name__ == "__main__":
    folder_path = ("visualization/reddit_simulation_align_with_human"
                   "/experiment_results")
    exp_name = "business_3600"
    db_path = folder_path + f"/{exp_name}.db"
    exp_info_file_path = folder_path + f"/{exp_name}.json"
    analysis_score.main(exp_info_file_path, db_path, exp_name, folder_path)
