import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split

pd.set_option("mode.chained_assignment", None)  # warning off


class TabularDataModule:
    def __init__(self, config: DictConfig):
        self.config = config

        self.train_data = None
        self.valid_data = None
        self.pred_frame_df = None

        self.prepare_data()
        self.setup()

    def prepare_data(self):
        """
        - load data file from csv file
        - Preprocess data
        """
        print(f"----------------- loading data : prepare_data() -----------------")
        self.tr_rating = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.train_file))
        self.tr_title = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.title_file), sep="\t")
        self.tr_year = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.year_file), sep="\t")
        self.tr_writer = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.writer_file), sep="\t")
        self.tr_genre = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.genre_file), sep="\t")
        self.tr_director = pd.read_csv(os.path.join(self.config.path.train_dir, self.config.path.director_file), sep="\t")

        # check and create preprocessed data
        files = os.listdir(self.config.path.train_dir)

        # check and create new version data (writer, director, year)
        if self.config.data.writers_version + self.config.path.writer_file not in files:
            print(">>> generating top writer columns ...")
            self.generate_top_writer()

        if self.config.data.dir_version + self.config.path.director_file not in files:
            print(">>> generating top director columns ...")
            self.generate_top_director()

        if self.config.data.years_version + self.config.path.year_file not in files:
            print(">>> filling out year columns ...")
            self.fillout_year()

        # read new version data (writer, director, year)
        v_writers = self.config.data.writers_version + self.config.path.writer_file
        self.v2_tr_writer = pd.read_csv(os.path.join(self.config.path.train_dir, v_writers), sep="\t")

        v_directors = self.config.data.dir_version + self.config.path.director_file
        self.v2_tr_director = pd.read_csv(os.path.join(self.config.path.train_dir, v_directors), sep="\t")

        v_years = self.config.data.years_version + self.config.path.year_file
        self.v2_tr_year = pd.read_csv(os.path.join(self.config.path.train_dir, v_years), sep="\t")

    def setup(self):
        """
        - Split data by user
        - Save datamoudle instance in variable
        """
        print(f"----------------- loading data : setup() -----------------")
        self.total_df = self.merge_total_data()

        # define important features
        self.user_col, self.item_col, self.target_col = "user", "item", "relevance"
        self.features = list(self.config.data.features)

        if self.config.trainer.cv_strategy == "kfold":
            self.train_kfold, self.valid_kfold = [], []
            splitter = GroupKFold(n_splits=self.config.trainer.kfold)

            for train_idx, valid_idx in splitter.split(self.total_df, groups=self.total_df["user"]):
                self.train_kfold.append(self.total_df.loc[train_idx])
                self.valid_kfold.append(self.total_df.loc[valid_idx])

            self.train_data = [TabularDataset(df, self.features, self.target_col) for df in self.train_kfold]
            self.valid_data = [TabularDataset(df, self.features, self.target_col) for df in self.valid_kfold]
        elif self.config.trainer.cv_strategy == "holdout":
            train, valid = train_test_split(self.total_df, test_size=0.2, random_state=42)

            self.train_data = TabularDataset(train, self.features, self.target_col)
            self.valid_data = TabularDataset(valid, self.features, self.target_col)
        else:
            raise Exception("Invalid cv strategy is entered")

        # create prediction frame df (pred_frame_df)
        self.pred_frame_df = self.tr_title.copy()
        self.pred_frame_df = pd.merge(self.pred_frame_df, self.v2_tr_year, on=["item"], how="outer")
        self.pred_frame_df = pd.merge(self.pred_frame_df, self.v2_tr_writer, on=["item"], how="outer")
        self.pred_frame_df = pd.merge(self.pred_frame_df, self.v2_tr_director, on=["item"], how="outer")
        onthot_genre = pd.get_dummies(self.tr_genre, columns=["genre"]).groupby("item").sum()
        self.pred_frame_df = pd.merge(self.pred_frame_df, onthot_genre, on=["item"], how="outer")

        self.pred_frame_df["year"] = self.pred_frame_df["writer"].astype("category")
        self.pred_frame_df["writer"] = self.pred_frame_df["writer"].astype("category")
        self.pred_frame_df["director"] = self.pred_frame_df["writer"].astype("category")

    def generate_top_writer(self):
        """
        - genereate new v2_writers.tsv file
        - only saves top writer of each movie item
        """
        rat_n_writer = pd.merge(self.tr_rating, self.tr_writer, how="outer", on=["item"])
        writer_ratio_df = pd.DataFrame(rat_n_writer["writer"].value_counts(), columns=["count"])
        writer_ratio_df = writer_ratio_df.reset_index()

        writer_cnt = pd.DataFrame({"item": [], "writer": [], "count": []})

        for item in self.tr_writer["item"].unique():
            tmp = self.tr_writer[self.tr_writer["item"] == item]
            if tmp.shape[0] >= 2:  # when writer is more than two
                tmp["count"] = tmp["writer"].apply(lambda x: writer_ratio_df[writer_ratio_df["writer"] == x]["count"].item())
                writer_cnt = pd.concat([writer_cnt, tmp], ignore_index=True)
            else:  # when writer is just one
                writer_cnt.loc[writer_cnt.shape[0]] = [
                    tmp["item"].item(),
                    tmp["writer"].item(),
                    writer_ratio_df[writer_ratio_df["writer"] == tmp["writer"].item()]["count"].item(),
                ]

        tr_top_writer = writer_cnt.loc[writer_cnt.groupby(["item"])["count"].idxmax()]

        v2_tr_writer = tr_top_writer.drop(["count"], axis=1)
        v2_tr_writer.to_csv(
            os.path.join(self.config.path.train_dir, self.config.data.writers_version + self.config.path.writer_file), sep="\t", index=False
        )

    def generate_top_director(self):
        """
        - genereate new v2_directors.tsv file
        - only saves top director of each movie item
        """
        rat_n_dir = pd.merge(self.tr_rating, self.tr_director, how="outer", on=["item"])
        dir_ratio_df = pd.DataFrame(rat_n_dir["director"].value_counts(), columns=["count"])
        dir_ratio_df = dir_ratio_df.reset_index()

        dir_cnt = pd.DataFrame({"item": [], "director": [], "count": []})

        for item in self.tr_director["item"].unique():
            tmp = self.tr_director[self.tr_director["item"] == item]
            if tmp.shape[0] >= 2:  # when director is more than two
                tmp["count"] = tmp["director"].apply(lambda x: dir_ratio_df[dir_ratio_df["director"] == x]["count"].item())
                dir_cnt = pd.concat([dir_cnt, tmp], ignore_index=True)
            else:  # when director is just one
                dir_cnt.loc[dir_cnt.shape[0]] = [
                    tmp["item"].item(),
                    tmp["director"].item(),
                    dir_ratio_df[dir_ratio_df["director"] == tmp["director"].item()]["count"].item(),
                ]

        tr_top_director = dir_cnt.loc[dir_cnt.groupby(["item"])["count"].idxmax()]

        v2_tr_director = tr_top_director.drop(["count"], axis=1)
        v2_tr_director.to_csv(
            os.path.join(self.config.path.train_dir, self.config.data.dir_version + self.config.path.director_file), sep="\t", index=False
        )

    def fillout_year(self):
        rat_year_title = pd.merge(self.tr_rating, self.tr_year, how="outer", on=["item"])
        rat_year_title = pd.merge(rat_year_title, self.tr_title, how="outer", on=["item"])

        def get_year_from_title(title):
            num_idx = title.rfind("(")
            if title[num_idx + 1 : num_idx + 5].isdigit():
                return int(title[num_idx + 1 : num_idx + 5])
            else:
                return 0

        null_year_df = pd.DataFrame(rat_year_title[rat_year_title["year"].isnull()]["item"], columns=["item"])
        null_year_df["year"] = rat_year_title[rat_year_title["year"].isnull()]["title"].apply(
            lambda x: get_year_from_title(x) if get_year_from_title(x) else np.NaN
        )
        null_year_df = null_year_df.drop_duplicates()

        final_year_df = pd.concat([self.tr_year, null_year_df])
        final_year_df.to_csv(
            os.path.join(self.config.path.train_dir, self.config.data.years_version + self.config.path.year_file), sep="\t", index=False
        )

    def generate_neg_sample(self, df: pd.DataFrame, sample_size=3) -> pd.DataFrame:
        """
        - generate negative sampling
        """
        print(">>> generating negative sampling ...")
        total_item_set = set(df["item"].unique())

        neg_samples = []
        for user in tqdm(df["user"].unique()):
            user_rated_item_set = set(df[df["user"] == user]["item"].unique())
            user_candidate_item_set = total_item_set - user_rated_item_set

            sample_size = len(user_rated_item_set)
            user_neg_samples_list = random.sample(user_candidate_item_set, sample_size)

            for neg_sample in user_neg_samples_list:
                neg_samples.append([user, neg_sample, 0])

        return pd.DataFrame(neg_samples, columns=["user", "item", "relevance"])

    def merge_total_data(self) -> pd.DataFrame:
        print(">>> merging whole dataframe to total_df ...")
        tr_rating_notime_df = self.tr_rating.drop(columns="time")
        tr_rating_notime_df["relevance"] = 1

        pos_samples_df = tr_rating_notime_df.copy()
        neg_samples_df = self.generate_neg_sample(tr_rating_notime_df)
        total_df = pd.concat([pos_samples_df, neg_samples_df])

        total_df = pd.merge(total_df, self.tr_title, on=["item"], how="outer")
        total_df = pd.merge(total_df, self.v2_tr_year, on=["item"], how="outer")
        total_df = pd.merge(total_df, self.v2_tr_writer, on=["item"], how="outer")
        total_df = pd.merge(total_df, self.v2_tr_director, on=["item"], how="outer")

        onthot_genre = pd.get_dummies(self.tr_genre, columns=["genre"]).groupby("item").sum()
        total_df = pd.merge(total_df, onthot_genre, on=["item"], how="outer")

        total_df["user"] = total_df["user"].astype("category")
        total_df["item"] = total_df["item"].astype("category")
        total_df["year"] = total_df["year"].astype("category")
        total_df["writer"] = total_df["writer"].astype("category")
        total_df["director"] = total_df["writer"].astype("category")

        return total_df


class TabularDataset:
    def __init__(self, df, features, target_col):
        self.X = df[features]
        self.y = df[target_col]
