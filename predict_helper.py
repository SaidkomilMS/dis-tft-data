"""Workaround for pytorch-forecasting trainer.predict() PredictCallback bug.

Manual prediction loop that calls model.forward() on each batch.
"""
import numpy as np
import torch


def predict_full(model, full_dl, full_ds):
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    all_preds = []
    for batch in full_dl:
        x = batch[0]
        x = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in x.items()}
        with torch.no_grad():
            out = model(x)
        all_preds.append(out["prediction"].cpu().numpy())
    out_arr = np.concatenate(all_preds, axis=0)
    decoder_idx = full_ds.x_to_index(full_ds)["time_idx"].values
    return out_arr, decoder_idx


def train_or_load_then_predict(
    target,
    name,
    params,
    df,
    META,
    ART_DIR,
    make_ds,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
    QuantileLoss,
    QUANTILES,
    pl,
    EarlyStopping,
    ModelCheckpoint,
    max_epochs=20,
):
    import os
    import json

    pred_npy = f"{ART_DIR}/preds_{name}.npy"
    idx_npy = f"{ART_DIR}/decoder_idx_{name}.npy"
    ckpt_path = f"{ART_DIR}/tft_{name}.ckpt"

    if os.path.exists(pred_npy) and os.path.exists(idx_npy):
        print(f"cached {name}")
        return np.load(pred_npy), np.load(idx_npy)

    train_ds = make_ds(target)
    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df, predict=False, stop_randomization=True)
    train_dl = train_ds.to_dataloader(train=True, batch_size=128, num_workers=2)
    val_dl = val_ds.to_dataloader(train=False, batch_size=256, num_workers=2)

    pl.seed_everything(7)

    if os.path.exists(ckpt_path):
        print(f"loading existing checkpoint for {name}")
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    else:
        print(f"training {name} from scratch")
        model = TemporalFusionTransformer.from_dataset(
            train_ds,
            **params,
            loss=QuantileLoss(quantiles=QUANTILES),
            log_interval=0,
        )
        es = EarlyStopping(monitor="val_loss", patience=4, mode="min")
        ckpt = ModelCheckpoint(
            monitor="val_loss", mode="min", dirpath=ART_DIR, filename=f"tft_{name}"
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            gradient_clip_val=0.5,
            callbacks=[es, ckpt],
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(model, train_dl, val_dl)

    full_ds = TimeSeriesDataSet.from_dataset(train_ds, df, predict=True, stop_randomization=True)
    full_dl = full_ds.to_dataloader(train=False, batch_size=256, num_workers=2)

    out_arr, decoder_idx = predict_full(model, full_dl, full_ds)
    np.save(pred_npy, out_arr)
    np.save(idx_npy, decoder_idx)
    return out_arr, decoder_idx
