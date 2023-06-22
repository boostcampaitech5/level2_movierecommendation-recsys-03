import os
import pandas as pd

if __name__ == "__main__":
    file_path = "./data/output/"

    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    except OSError:
        print("Error: Creating directory. " + file_path)

    # write file name
    file1_name = "ease_500_10.csv"
    file2_name = "18_113755_MultiVAE_submit.csv"

    file1_df = pd.read_csv(os.path.join(file_path, file1_name))
    file2_df = pd.read_csv(os.path.join(file_path, file2_name))

    # first priority file
    file1_df["tem"] = 2
    file1_df["seq"] = file1_df.groupby("user")["tem"].apply(lambda x: x.cumsum()).droplevel(axis=0, level=0)
    file1_df["seq"] = file1_df["seq"] - 1

    # second priority file
    file2_df["tem"] = 2
    file2_df["seq"] = file2_df.groupby("user")["tem"].apply(lambda x: x.cumsum()).droplevel(axis=0, level=0)

    final = pd.concat([file1_df, file2_df])

    # ensemble
    final["seq"][final.duplicated(["user", "item"], keep=False)] = 0  # Top Prioritiy what appears in common
    final = final.drop_duplicates(["user", "item"], keep="first")  # Deduplication
    final = final.sort_values(["user", "seq"]).reset_index(drop=True)
    final = final.groupby("user").apply(lambda x: x[:10]).reset_index(drop=True)

    file_name = "ens_" + file1_name[:-4] + "_" + file2_name[:-4] + ".csv"
    print(f">>> ensemble is finished! : {file_name}")

    final[["user", "item"]].to_csv(os.path.join(file_path, file_name), index=False)
