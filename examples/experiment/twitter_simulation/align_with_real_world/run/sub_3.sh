folder_path="scripts/twitter_simulation/align_with_real_world/yaml_200/sub3"
# Read all filenames in the folder
for file in "$folder_path"/*; do
    # Extract the filename (excluding the extension)
    filename=$(basename "$file")
    topicname="${filename%.*}"

    # Generate paths for csv and db files
    # config_path = "${folder_path}/${topicname}.yaml"

    # Run the python script
    # python main.py --config "${folder_path}/${topicname}.yaml"
    python scripts/twitter_simulation/twitter_simulation_large.py --config_path "${folder_path}/${topicname}.yaml"
    # python visualization/result_ana.py --topic_name "$topicname"
done
