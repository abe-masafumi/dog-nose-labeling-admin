# 🐕 犬の鼻ラベリング管理ツール

犬の顔画像から「犬の鼻が含まれているか」を判定し、ラベルを付与・管理するためのWebベースのラベリング管理ツールです。機械学習用データセットの作成を効率的に行うことができます。

## 📋 機能概要

### 主要機能
- **画像表示・ラベリング**: 1枚ずつ画像を表示し、直感的なGUIでラベル付与
- **キーボードショートカット**: 効率的なラベリング作業をサポート
- **ラベル管理**: SQLiteデータベースによる確実なデータ保存
- **データエクスポート**: JSON/CSV形式でのラベル結果出力
- **データセット分割**: 機械学習用のtrain/val/test分割対応
- **フォルダ分割出力**: 機械学習フレームワーク用の自動フォルダ構造生成

### ラベル種別
- **メインラベル**
  - `nose`: 犬の鼻が含まれている画像
  - `non_nose`: 犬の鼻が含まれていない画像

- **サブラベル**（任意）
  - 鼻の向き: `front`（正面）、`side`（横向き）、`tilted`（斜め）
  - 鼻の状態: `wet`（濡れている）、`dry`（乾いている）、`blurred`（ぼやけている）

## 🚀 セットアップ

### 必要環境
- Python 3.7以上
- Flask 2.3.3以上

### インストール手順

1. **リポジトリのクローン**
```bash
git clone https://github.com/abe-masafumi/dog-nose-labeling-admin.git
cd dog-nose-labeling-admin
```

2. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

3. **画像フォルダの準備**
```bash
mkdir images
# imagesフォルダに犬の顔画像（JPEG/PNG形式）を配置
```

4. **アプリケーションの起動**
```bash
python app.py
```

5. **ブラウザでアクセス**
```
http://localhost:5000
```

## 📁 プロジェクト構造

```
dog-nose-labeling-admin/
├── app.py                 # メインアプリケーション
├── templates/
│   └── index.html        # Webインターフェース
├── images/               # 画像ファイル格納フォルダ
├── labels.db            # SQLiteデータベース（自動生成）
├── dataset/             # エクスポート用フォルダ（自動生成）
├── requirements.txt     # Python依存関係
└── README.md           # このファイル
```

## 🎯 使用方法

### 基本的なラベリング手順

1. **画像の配置**: `images/`フォルダに犬の顔画像を配置
2. **アプリケーション起動**: `python app.py`でサーバーを開始
3. **ラベリング作業**:
   - 画像が自動的に表示されます
   - メインラベル（鼻あり/鼻なし）を選択
   - 必要に応じてサブラベルを追加
   - データセット分割（train/val/test）を指定
   - 「ラベルを保存」ボタンで保存

### キーボードショートカット

| キー | 機能 |
|------|------|
| `1` | 鼻ありラベル |
| `2` | 鼻なしラベル |
| `F` | 正面向き |
| `S` | 横向き |
| `T` | 斜め向き |
| `W` | 濡れている |
| `D` | 乾いている |
| `B` | ぼやけている |
| `←` | 前の画像 |
| `→` | 次の画像 |
| `Enter` | ラベル保存 |

### データエクスポート

#### JSON/CSV形式
- 「JSON」または「CSV」ボタンでラベル結果をダウンロード
- ファイル名: `labels_export_YYYYMMDD_HHMMSS.json/csv`

#### データセット分割出力
- 「📁 データセット出力」ボタンで機械学習用フォルダ構造を生成
- 出力構造:
```
dataset/
├── train/
│   ├── nose/
│   └── non_nose/
├── val/
│   ├── nose/
│   └── non_nose/
└── test/
    ├── nose/
    └── non_nose/
```

## 🗄️ データベース構造

### labelsテーブル
| カラム | 型 | 説明 |
|--------|----|----|
| id | INTEGER | 主キー |
| image_path | TEXT | 画像ファイルパス |
| main_label | TEXT | メインラベル（nose/non_nose） |
| sub_labels | TEXT | サブラベル（JSON形式） |
| dataset_split | TEXT | データセット分割（train/val/test） |
| created_at | TIMESTAMP | 作成日時 |
| updated_at | TIMESTAMP | 更新日時 |

### imagesテーブル
| カラム | 型 | 説明 |
|--------|----|----|
| id | INTEGER | 主キー |
| filename | TEXT | ファイル名 |
| filepath | TEXT | ファイルパス |
| file_size | INTEGER | ファイルサイズ |
| created_at | TIMESTAMP | 作成日時 |

## 🔧 API エンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/` | GET | メインページ |
| `/api/images` | GET | 全画像情報取得 |
| `/api/images/<id>` | GET | 特定画像情報取得 |
| `/api/labels` | POST | ラベル保存 |
| `/api/export/json` | GET | JSON形式エクスポート |
| `/api/export/csv` | GET | CSV形式エクスポート |
| `/api/export/dataset` | GET | データセット分割出力 |
| `/images/<filename>` | GET | 画像ファイル配信 |

## 🔮 将来的な拡張予定

- **半自動ラベリング支援**: 機械学習モデルによる推定ラベル提示
- **複数ユーザー対応**: 同時ラベリング・権限管理
- **アクティブラーニング**: 曖昧なデータの優先提示
- **クラウドストレージ連携**: S3/Firebase対応
- **アノテーションログ**: 作業履歴の詳細管理

## 🤝 コントリビューション

プルリクエストやイシューの報告を歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 👤 作成者

- **安倍将史** ([@abe-masafumi](https://github.com/abe-masafumi))
- **Devin AI** - 実装支援

---

**Link to Devin run**: https://app.devin.ai/sessions/a157972900e64fc0966e757cf8f7a9c5
