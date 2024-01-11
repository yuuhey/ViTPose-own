### MAE Pre-trained model
---------
mae_pretrained/ 디렉토리 내 pretrained model 저장

The small size MAE pre-trained model can be found in [Onedrive.](https://onedrive.live.com/?authkey=%21AHohY4eAye4I2Mo&id=E534267B85818129%2125497&cid=E534267B85818129&parId=root&parQt=sharedby&o=OneUp)
The base, large, and huge pre-trained models using MAE can be found in the [MAE official repo.](https://github.com/facebookresearch/mae)

scripts example
```
bash tools/dist_train.sh configs/body/2d_kpt_sview_rgb_imgll_coco_256x192.py 2 --cfg-options model.pretrained=mae_pretrained/mae_small_pretrain.pth --seed 0 --suffix 1
```
