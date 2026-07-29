#!/usr/bin/env python
# Author: Alissa Valentine 2025

# import packages
import argparse
import math
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tabtransformertf.models.fttransformer import FTTransformer, FTTransformerEncoder
from tabtransformertf.utils.preprocessing import df_to_dataset
from tabtransformertf.utils.helper import get_model_importances
from datetime import datetime


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

def create_model_dir(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory for model logging, checkpoints, etc: {model_dir}")
    else:
        print(f"Warning: {model_dir} already exists. You might overwrite existing files.")

class WarmUpLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, warmup_steps, total_steps, start_lr=2e-5):
        super().__init__()
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        warmup_lr = self.start_lr + (self.max_lr - self.start_lr) * (step / self.warmup_steps)
        decay_lr = self.max_lr * (1 - (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        return tf.maximum(lr, 0.0)
    
    def get_config(self):
        return{
            "max_lr": self.max_lr,
            "start_lr": self.start_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps
        }

def test_model_region(train_region, test_region, transformer, test_raw, test_processed, features, model_path_dir):
    # test on region
    print(f"Testing on {test_region} data, saving test results and importances")
    periodic_test_preds = transformer.predict(test_processed, verbose=2)
    test_raw['y_prob'] = periodic_test_preds['output'].ravel()
    test_raw['y_pred'] = periodic_test_preds['output'].ravel()>0.5
    test_raw['y_true'] = test_raw["MDDdx"]==1

    print("Test ROC AUC:", np.round(roc_auc_score(test_raw['y_true'], test_raw['y_prob']), 4))
    print("Test PR AUC:", np.round(average_precision_score(test_raw['y_true'], test_raw['y_prob']), 4))
    print("Test Accurary:", np.round(accuracy_score(test_raw['y_true'], test_raw['y_pred']), 4))

    # save prediction scores
    test_raw.to_csv(f"{model_path_dir}test_{test_region}_train_{train_region}_pMDDx.csv")

    # save feature importances
    print(f"\n================Saving model feature importance for {test_region}================")
    periodic_importances_df = pd.DataFrame(periodic_test_preds['importances'][:, :-1], columns = features)
    periodic_importances_df.to_csv(f"{model_path_dir}test_{test_region}_train_{train_region}_importances.csv")
    # periodic_total_importances = get_model_importances(
    #     periodic_importances_df, title = f"Importances for FT-Transformer with Periodic Numerical Embeddings trained on {train_region} tested on {test_region}"
    # )

    # plt.figure(figsize=(15,7))
    # ax = periodic_total_importances.plot.bar()
    # for p in ax.patches:
    #     ax.annotate(str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height()*1.01))
    # plt.title(f"Importances for FT-Transformer with Periodic Numerical Embeddings trained on {train_region} tested on {test_region}")
    # plt.savefig(f"{model_path_dir}test_{test_region}_train_{train_region}_average_importances.png")
    # periodic_total_importances.to_csv(f"{model_path_dir}test_{test_region}_train_{train_region}_average_importances.csv")

    # explore max and min prediction
    # largest prediction
    # max_idx = np.argsort(periodic_test_preds['output'].ravel())[-1]
    # example_importance_max = periodic_importances_df.iloc[max_idx, :].sort_values(ascending=False).rename("Importance").to_frame().join(test_raw.iloc[max_idx, :].rename("Example Values"))
    # print(f"The contributions to row {max_idx} which was scored {str(np.round(periodic_test_preds['output'].ravel()[max_idx], 4))}")
    # print(example_importance_max)

    # # smallest prediction
    # min_idx = np.argsort(periodic_test_preds['output'].ravel())[0]
    # example_importance_min = periodic_importances_df.iloc[min_idx, :].sort_values(ascending=False).rename("Importance").to_frame().join(test_raw.iloc[min_idx, :].rename("Example Values"))
    # print(f"The contributions to row {min_idx} which was scored {str(np.round(periodic_test_preds['output'].ravel()[min_idx], 4))}")
    # print(example_importance_min)

    # example_importance_max.to_csv(f"{model_path_dir}test_{test_region}_train_{train_region}_example_importances_max.csv")
    # example_importance_min.to_csv(f"{model_path_dir}test_{test_region}_train_{train_region}_example_importances_min.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program fine-tunes and evaluates a FTTransformer model using periodic embedding and relu activation.')
    parser.add_argument('--dataset_path', help='provide the file path to the dataset', default='cleaned_dataset.csv')
    parser.add_argument('--models_path', help='provide the file path to the models', default='models/')
    parser.add_argument('--model_type', help='would you like to use ideal or bad variables?', default='ideal')
    parser.add_argument('--train_region', help='provide the name of the region for finetuning: Hovedstaden (Capital/Copenhagen), Sjaelland (Zealand), Nordjylland, Midtjylland, or Syddanmark', default='Hovedstaden')
    parser.add_argument('--seed', help='set the seed', default=42, type=int)
    parser.add_argument('--epoch_count', help='set epoch count for model finetuning', default=3, type=int)
    parser.add_argument('--early_stop_patience', help='set patience for model early stopping', default=5, type=int)
    parser.add_argument('--learning_rate', help='set learning rate (or max if using scheduler) for model finetuning', default=0.0001, type=float)
    parser.add_argument('--lr_scheduler', help='set True if using learning rate scheduler', default=False, type=bool)
    parser.add_argument('--weight_decay', help='set weight decay for model finetuning', default=0.0001, type=float)
    parser.add_argument('--batch_size', help='set batch size for model finetuning', default=521, type=int)
    args = parser.parse_args()

    # set seeds
    set_all_seeds(args.seed)

    # set datetime
    run_date = datetime.today().strftime("%d-%m-%Y")

    # set model path
    model_path = f"{args.models_path}{args.train_region}_sigmoid_epoch-{args.epoch_count}_lr-{args.learning_rate}scheduled_patience-{args.early_stop_patience}_{run_date}/"
    create_model_dir(model_path)

    # load data
    print("\n================Loading data================")
    dataset = pd.read_csv(args.dataset_path)

    # set categorical and numerical variables
    if args.model_type == 'bad':
        NUMERIC = ["age_use", "income_rank"]
        CATEGORICAL = ["sex", "western_danish", "status_civ", "HUDD_inferred", "region", "komgroup"]
    else:
        NUMERIC = ["age_use", "moves", "income_rank"]
        CATEGORICAL = ["sex", "LGBT", "livingalone", "danishborn", "status_civ", "region", "komgroup", "HUDD_inferred", "UNIntermediateRegion", "WorldBankLevel"]

    FEATURES = NUMERIC + CATEGORICAL
    LABEL = "MDDdx"

    dataset[CATEGORICAL] = dataset[CATEGORICAL].astype(str)
    dataset[NUMERIC] = dataset[NUMERIC].astype(float)

    # split the train and test sets
    train_region = dataset[dataset.region == args.train_region]
    train = train_region[train_region.subpop == 1]
    test = train_region[train_region.subpop == 2]
    X_test, X_val = train_test_split(test, test_size=0.4, random_state=args.seed)
    print(f"Data used in this experiment:\nThere are {len(train)} pts in the training set\nThere are {len(X_val)} pts in the validation set\nThere are {len(X_test)} pts in the test set")

    # create the test sets
    test_regions = ["Hovedstaden", "Sjaelland", "Nordjylland", "Midtjylland", "Syddanmark"]
    test_regions.remove(args.train_region)
    test_1 = dataset[dataset.region == test_regions[0]]
    test_2 = dataset[dataset.region == test_regions[1]]
    test_3 = dataset[dataset.region == test_regions[2]]
    test_4 = dataset[dataset.region == test_regions[3]]

    # convert df to dataset object type
    print("\n================Preprocessing data================")
    sc = StandardScaler()
    train.loc[:, NUMERIC] = sc.fit_transform(train[NUMERIC])
    X_val.loc[:, NUMERIC] = sc.fit_transform(X_val[NUMERIC])
    X_test.loc[:, NUMERIC] = sc.fit_transform(X_test[NUMERIC])
    test_1.loc[:, NUMERIC] = sc.fit_transform(test_1[NUMERIC])
    test_2.loc[:, NUMERIC] = sc.fit_transform(test_2[NUMERIC])
    test_3.loc[:, NUMERIC] = sc.fit_transform(test_3[NUMERIC])
    test_4.loc[:, NUMERIC] = sc.fit_transform(test_4[NUMERIC])

    train_dataset = df_to_dataset(train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)
    test_dataset_0 = df_to_dataset(X_test[FEATURES + [LABEL]], LABEL, shuffle=False)
    test_dataset_1 = df_to_dataset(test_1[FEATURES + [LABEL]], LABEL, shuffle=False)
    test_dataset_2 = df_to_dataset(test_2[FEATURES + [LABEL]], LABEL, shuffle=False)
    test_dataset_3 = df_to_dataset(test_3[FEATURES + [LABEL]], LABEL, shuffle=False)
    test_dataset_4 = df_to_dataset(test_4[FEATURES + [LABEL]], LABEL, shuffle=False)

    # initialize periodic encoder
    print("\n================Initializing Encoder================")
    ft_periodic_encoder = FTTransformerEncoder(
        numerical_features=NUMERIC,
        categorical_features=CATEGORICAL,
        numerical_data=train[NUMERIC].values,
        categorical_data=train[CATEGORICAL].values,
        y = None,
        numerical_embedding_type='periodic',
        numerical_bins=128,
        embedding_dim=16,
        depth=4,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.2,
        explainable=True
    )
    
    # initialize transformer
    print("\n================Initializing Transformer================")
    ft_periodic_transformer = FTTransformer(
        encoder=ft_periodic_encoder,
        out_dim=1,
        out_activation='sigmoid'
    )

    if args.lr_scheduler:
        print("\n================Initializing Learning Rate Scheduler================")
        # initialize learning rate scheduler
        total_steps = int((len(train) / args.batch_size) * args.epoch_count)
        warmup_steps = int(0.1 * total_steps)

        lr_schedule = WarmUpLinearDecay(
            max_lr=args.learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            start_lr=2e-5
        )

        # include adamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate = lr_schedule,
            weight_decay = args.weight_decay
        )
    else:
        # include adamW optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate = args.learning_rate,
            weight_decay = args.weight_decay
        )

    # include adamW optimizer
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay
    )

    # set loss and metric functions
    ft_periodic_transformer.compile(
        optimizer = optimizer,
        loss = {'output': tf.keras.losses.BinaryCrossentropy(), 'importances': None},
        metrics = {'output': [tf.keras.metrics.AUC(name="ROC_AUC", curve='ROC')], 'importances':None}
    )

    # create callback list to include early stopping, logging, and checkpointing
    early = EarlyStopping(monitor='val_output_loss',
                        mode='min',
                        patience=args.early_stop_patience,
                        restore_best_weights=True)

    csv_logger = CSVLogger(f'{model_path}training_log.csv',
                        append=False)

    checkpoint_path = f"{model_path}" + "ckpt/{epoch:02d}-{val_loss:.2f}.weights.h5"

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_output_loss",
        save_best_only=True,
        mode="min",
        save_weights_only=True
    )

    callback_list = [early, csv_logger, model_checkpoint]

    # fine tune the model!
    print("\n================Fine tuning the model with checkpointing, logging, and early stopping...================")
    ft_periodic_history = ft_periodic_transformer.fit(
        train_dataset,
        epochs = args.epoch_count,
        validation_data = val_dataset,
        callbacks=callback_list,
        verbose=2
    )

    # save full model
    print("\n================Saving final model================")
    ft_periodic_transformer.save(f"{model_path}final_model_save", save_format='tf')

    # plot model performance
    print("\n================Plotting model performance================")
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(ft_periodic_history.history['loss'], label='Training Loss')
    ax[0].plot(ft_periodic_history.history['val_loss'], label='Validation Loss')
    ax[0].legend()
    ax[1].plot(ft_periodic_history.history['output_ROC_AUC'], label='Training ROC AUC')
    ax[1].plot(ft_periodic_history.history['val_output_ROC_AUC'], label='Validation ROC AUC')
    ax[1].legend()
    fig.savefig(f"{model_path}train_valid_loss_ROC.png")

    # Test model
    print("\n================Testing model on each region================")
    print(f"Testing FT-Transformer with Periodic Embedding finetuned on {args.train_region}")

    test_model_region(args.train_region, args.train_region, ft_periodic_transformer, X_test, test_dataset_0, FEATURES, model_path)
    test_model_region(args.train_region, test_regions[0], ft_periodic_transformer, test_1, test_dataset_1, FEATURES, model_path)
    test_model_region(args.train_region, test_regions[1], ft_periodic_transformer, test_2, test_dataset_2, FEATURES, model_path)
    test_model_region(args.train_region, test_regions[2], ft_periodic_transformer, test_3, test_dataset_3, FEATURES, model_path)
    test_model_region(args.train_region, test_regions[3], ft_periodic_transformer, test_4, test_dataset_4, FEATURES, model_path)

    print("\n================Script completed!================")