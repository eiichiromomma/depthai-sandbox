# OAK-DのSandbox

Apple Siliconで唯一安定して使える状態のRGB-Dカメラシリーズ用の色々

## 導入

```bash
git clone https://github.com/luxonis/depthai-python.git
cd depthai-python
python3 -m venv ENV
source ENV/bin/activate
python3 -m pip install -U pip ipython "numpy<2"
cd examples
python3 install_requirements.py
```

で一通りインストールされてdepthai-pythonのサンプルは動くようになる。

落ちモノインタラクションのBouncyBallsDepthAI.pyはpymunkとpygameが必要なので
```bash
python3 -m pip install pymunk pygame
```

## 作ったもの

* depth_post_processing: depthai-aiのサンプルを弄って特定距離範囲を2値化で表示する
* depth_confidence_post_processing: depthとconfidenceMapを取得する例
* BouncyBallsDepthAI: 落ちモノインタラクション
