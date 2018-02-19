# error_detection
Bi-LSTMを使い日本語の文単位の誤り検出を行う。

```
python BLSTM.py train # モデルを学習し、コード内で指定したディレクトリにモデルを保存
```

入力：

入力となるコーパスは、１行にラベルと分かち書きされた文がタブ区切りで書かれている。

（例）0\t私 は 食べる 。


ラベルは、0が正文で1が非文である。

出力：

Precision,Recall,F-value,Accuracyが出力される。


```
python BLSTM.py test # モデルをテストする
```
