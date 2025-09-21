pip install -r requirements.txt
dog-nose-labeling-admin/
dataset/

# 🐕 犬の鼻ラベリング管理ツール

犬の顔画像から「犬の鼻が含まれているか」を判定し、ラベルを付与・管理するためのWebベースのラベリング管理ツールです。機械学習用データセットの作成を効率的に行えます。

---

## 📋 主な機能

- **画像ラベリング**: 画像を1枚ずつ表示し、直感的なGUIでラベル付与・編集
- **レビュー画面**: フィルタ・検索・未作業絞り込み・一括確認
- **重複画像検出・削除**: sha256/pHashによる重複・類似画像グループ化と論理削除
- **自動鼻検出バッチ**: YOLOv8モデルによる鼻bbox自動付与（コマンド実行）
- **データセット自動分割**: train/val/test比率・詳細ターゲット指定で自動分割
- **データエクスポート**: JSON/CSV/YOLO形式・分割済みフォルダ・個別ZIP出力
- **論理削除**: 画像・ラベルはDB上で論理削除、全画面で除外
- **キーボードショートカット**: 効率的なラベリング作業

---

## 🚀 セットアップ

### 必要環境
- Python 3.7以上
- Flask 2.3.3以上

### インストール手順
```bash
git clone https://github.com/abe-masafumi/dog-nose-labeling-admin.git
cd dog-nose-labeling-admin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir images
# images/ に犬の顔画像（JPEG/PNG）を配置
```

### アプリ起動
```bash
python app.py
# ブラウザで http://localhost:8080 を開く
```

---

## �️ 画面構成

- **ラベリング** `/` : 画像1枚ずつラベル付与・bbox編集・保存・個別エクスポート
- **レビュー** `/review` : フィルタ・検索・未作業絞り込み・一括確認
- **エクスポート** `/export` : JSON/CSV/YOLO/分割済みデータセット出力・自動分割
- **重複確認** `/duplicates` : sha256/pHashによる重複・類似画像グループ化と論理削除
- **使用方法** `/usage` : 操作ガイド・ショートカット一覧

---

## 🎯 ラベリング・レビュー手順

1. `images/` フォルダに画像を配置
2. `python app.py` でサーバー起動
3. `/` で画像を1枚ずつラベリング
   - メインラベル（鼻あり/なし）・サブラベル・分割・bboxを設定
   - 「ラベルを保存」でDBに反映
4. `/review` でフィルタ・検索・未作業絞り込み・一括確認
5. `/duplicates` で重複・類似画像を検出し、不要画像を論理削除
6. `/export` でデータセット出力・自動分割

---

## 🏷️ ラベル仕様

- **メインラベル**: `nose`（鼻あり）/ 未設定（鼻なし）
- **サブラベル**: 向き（front/side/tilted）、鮮明度（clear/blurred）、色（black/brown/gray/pink/marble）、大きさ（large/small）、毛色（light_fur/dark_fur）、鼻の長さ（nose_long/nose_medium/nose_short）
- **分割**: train/val/test
- **bbox**: {x_min, y_min, x_max, y_max} 形式

---

## 🤖 YOLO自動鼻検出バッチ

YOLOv8モデルで未手動修正画像に自動で鼻bboxを付与できます。

### 使い方
1. `models/8_30_best.pt` を配置
2. `python app.py detect_nose` を実行

> is_manual=1（手動修正済み）は上書きしません。bboxは {x_min, y_min, x_max, y_max} で保存。main_labelも自動更新。is_completedは自動バッチでは変更しません。

---

## 🔁 重複画像検出・論理削除

- `/duplicates` 画面でsha256/pHashによる重複・類似画像グループを検出
- UI上で不要画像を論理削除（DB上でdeleted_atセット、全画面で除外）
- 削除後は再検出せずUIのみ即時反映

---

## 📤 データエクスポート

- `/export` 画面で以下が可能
  - JSON/CSV形式で全ラベル出力
  - YOLO形式（画像＋ラベルtxt）で個別/一括ZIP出力
  - train/val/test分割済みフォルダ構造で出力
  - **自動分割**: 比率・詳細ターゲット指定で分割、プレビュー・実行

---

## 🗄️ データベース構造

### labelsテーブル
| カラム         | 型        | 説明 |
|----------------|-----------|----------------------------|
| id             | INTEGER   | 主キー                     |
| image_path     | TEXT      | 画像ファイルパス（ユニーク）|
| main_label     | TEXT      | メインラベル               |
| sub_labels     | TEXT      | サブラベル（JSON形式）     |
| dataset_split  | TEXT      | データセット分割           |
| bbox           | TEXT      | バウンディングボックス     |
| is_completed   | INTEGER   | 作業済みフラグ             |
| is_manual      | TEXT      | 手動/自動/未設定           |
| deleted_at     | TIMESTAMP | 論理削除日時               |
| created_at     | TIMESTAMP | 作成日時                   |
| updated_at     | TIMESTAMP | 更新日時                   |

### imagesテーブル
| カラム | 型 | 説明 |
|--------|----|----|
| id | INTEGER | 主キー |
| filename | TEXT | ファイル名 |
| filepath | TEXT | ファイルパス |
| file_size | INTEGER | ファイルサイズ |
| deleted_at | TIMESTAMP | 論理削除日時 |
| created_at | TIMESTAMP | 作成日時 |

---

## 🔧 API エンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/` | GET | ラベリング画面 |
| `/review` | GET | レビュー画面 |
| `/export` | GET | エクスポート画面 |
| `/duplicates` | GET | 重複確認画面 |
| `/usage` | GET | 使用方法ガイド |
| `/api/images` | GET | 全画像情報取得 |
| `/api/images/<id>` | GET | 特定画像情報取得 |
| `/api/labels` | POST | ラベル保存 |
| `/api/export/json` | GET | JSON形式エクスポート |
| `/api/export/csv` | GET | CSV形式エクスポート |
| `/api/export/dataset` | GET | データセット分割出力 |
| `/api/export_single/<id>` | GET | 画像1枚分のYOLOデータセットZIP |
| `/api/duplicates` | GET | 重複/類似画像グループ取得 |
| `/api/duplicates/delete` | POST | 画像の論理削除 |
| `/api/auto-split` | POST | データセット自動分割実行 |
| `/api/auto-split/settings` | GET/POST | 自動分割設定取得・保存 |
| `/images/<filename>` | GET | 画像ファイル配信 |

---

## ⌨️ キーボードショートカット

| キー | 機能 |
|------|------|
| `1` | 鼻ありラベル |
| `2` | 鼻なしラベル |
| `F` | 正面向き |
| `S` | 横向き |
| `T` | 斜め向き |
| `B` | ぼやけ |
| `C` | 鮮明 |
| `K` | 黒 |
| `R` | 茶色 |
| `G` | 灰色 |
| `P` | ピンク |
| `M` | マーブル |
| `L` | 大きい |
| `Q` | 小さい |
| `J` | 明るい毛色 |
| `N` | 暗い毛色 |
| `O` | 長い鼻 |
| `Shift+M` | 中くらい鼻 |
| `Shift+S` | 短い鼻 |
| `←` | 前の画像 |
| `→` | 次の画像 |
| `Enter` | ラベル保存 |


python main.py sample.jpg "0 0.190278 0.200391 0.183333 0.103125"   

