folder_path="scripts/twitter_simulation/align_with_real_world/yaml_200/sub6"
# 读取文件夹中的所有文件名
for file in "$folder_path"/*; do
    # 提取文件名（不包括后缀）
    filename=$(basename "$file")
    topicname="${filename%.*}"

    # 生成csv和db文件路径
    # config_path = "${folder_path}/${topicname}.yaml"

    # 运行python脚本
    # python main.py --config "${folder_path}/${topicname}.yaml"
    python scripts/twitter_simulation/twitter_simulation_large.py --config_path "${folder_path}/${topicname}.yaml"
    # python visualization/result_ana.py --topic_name "$topicname"
done
