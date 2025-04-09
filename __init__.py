
if __name__ == '__main__':
    import os
    import  csv

    ppo_cnn_log= "logs/ppo_cnn.csv"
    os.makedirs(os.path.dirname(ppo_cnn_log), exist_ok=True)

    ppo_vit_log = "logs/ppo_vit.csv"
    os.makedirs(os.path.dirname(ppo_vit_log), exist_ok=True)

    with open(ppo_cnn_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_reward"])

    with open(ppo_vit_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_reward"])