# Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

## はじめに

このリポジトリは[ST-GCN](https://github.com/yysijie/st-gcn)を元にオリジナルデータで学習するためのコンフィグファイルを追加し、姿勢推定モデルをOpenPoseからYOLOv7へ変更したものです。

## 元論文

**Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition** Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

## インストール

```bash
$ git clone -b st-gcn https://github.com/Jeong-Labo/st-gcn.git
$ pip install -r requirements.txt
$ cd torchlight
$ python setup.py install
$ cd ../
```

`No module named 'torchlight'`と出た時は、現状は手動コピーで対処するしかない。

GPU版torch、torchvisionを利用する際は、[PyTorch公式サイト](https://pytorch.org/)のインストール方法に従うこと。

一応Docker上で動作させることもできる。その際もホストOS側で次のモデルのダウンロードはしておくこと。

## 学習済みモデルのダウンロード

- [YOLOv7-Pose](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)
- [ST-GCN](https://drive.google.com/open?id=1MDIUlJ_X-IpNgLZYXgdMuclCsnYTepU1&authuser=0&usp=drive_link)

この2つをダウンロードし、weights/内に移動しておく。

## デモ

- 通常デモ
```bash
$ python main.py demo_offline --weights ./weights/st_gcn.kinetics.pt --video ${PATH_TO_VIDEO}
```

- リアルタイムデモ(Docker不可)
```bash
$ python main.py demo --weights ./weights/st_gcn.kinetics.pt --video ${PATH_TO_VIDEO}
```

## 学習

学習用のJSONデータを用意し、これをNumPy形式に変換する。
```bash
python tools/maborosi_gendata.py --data_path ${PATH_TO_DATA_DIR}
```

学習
```bash
python main.py recognition -c config/st_gcn/${DATASET}/train.yaml
```

## テスト

```bash
python main.py recognition -c config/st_gcn/${DATASET}/test.yaml --weights ${PATH_TO_WEIGHTS}
```

## ST-GCNの可視化結果のサンプル

出力画面

<p align="center">
    <img src="resource/info/demo_video.gif", width="1200">
</p>

各クラス(一部抜粋)の出力例

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="resource/info/S001C001P001R001A044_w.gif"></td>
    <td><img width="150px" src="resource/info/S003C001P008R001A008_w.gif"></td>
    <td><img width="150px" src="resource/info/S002C001P010R001A017_w.gif"></td>
    <td><img width="150px" src="resource/info/S003C001P008R001A002_w.gif"></td>
    <td><img width="150px" src="resource/info/S001C001P001R001A051_w.gif"></td>
  </tr>
  <tr>
    <td><font size="1">Touch head<font></td>
    <td><font size="1">Sitting down<font></td>
    <td><font size="1">Take off a shoe<font></td>
    <td><font size="1">Eat meal/snack<font></td>
    <td><font size="1">Kick other person<font></td>
  </tr>
  <tr>
    <td><img width="150px" src="resource/info/hammer_throw_w.gif"></td>
    <td><img width="150px" src="resource/info/clean_and_jerk_w.gif"></td>
    <td><img width="150px" src="resource/info/pull_ups_w.gif"></td>
    <td><img width="150px" src="resource/info/tai_chi_w.gif"></td>
    <td><img width="150px" src="resource/info/juggling_balls_w.gif"></td>
  </tr>
  <tr>
    <td><font size="1">Hammer throw<font></td>
    <td><font size="1">Clean and jerk<font></td>
    <td><font size="1">Pull ups<font></td>
    <td><font size="1">Tai chi<font></td>
    <td><font size="1">Juggling ball<font></td>
  </tr>
</table>
