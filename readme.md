# vmlkit の使い方

## モジュール

- const.py
- utility.py
- validator.py
- visualizer.py
- preprocessing
  - feature_selector.py
  - optimizer.py
  - sampler.py
  - tuneupper.py
- model_selection
  - analyzer.py
  - cleaner.py
  - encoder.py
  - preprocessor.py
  - scaler.py

## 実行ファイル

1. config.py：入出力ファイルのパスやパラメータなどを設定
2. preprocess.py：データの前処理を実行
3. optimize.py：最適な機械学習モデルと重要な特徴量を選択
4. train.py：optimize.py で選ばれたモデルをさらに最適化（任意）
5. test.py：提出用csvファイルを作成
