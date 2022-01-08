# 自然言語処理に用いるツール

## 形態素解析

### MeCab

* <https://taku910.github.io/mecab/>
* 言語、辞書、コーパスに依存しない汎用的な設計を基本方針としている
* 非常に高速に動作する
* 単語の発生しやすさ(生起コスト)と品詞の繋がりやすさ(連接コスト)から、コストが最も小さくなる解析結果を出力する

### JUMAN

* <http://nlp.ist.i.kyoto-u.ac.jp/index.php?JUMAN>
* 使用者によって文法の定義、単語間の接続関係の定義などを容易に変更できるように配慮している
* [JUMANのデモ](http://lotus.kuee.kyoto-u.ac.jp/nl-resource/cgi-bin/juman.cgi)

### JUMAN++

* <http://nlp.ist.i.kyoto-u.ac.jp/index.php?JUMAN++>
* RNNLMを使用することで意味的な自然さを考慮した解析を行える
* 精度は高いが動作速度は遅い
* 現在のところ、開発版である
* [JUMAN++のデモ](http://tulip.kuee.kyoto-u.ac.jp/demo/jumanpp_lattice)

### Sudachi

* <https://github.com/WorksApplications/Sudachi>
* 複数の分割単位の併用、文字正規化や未知語処理に機能追加が可能といった特徴を持つ

### Janome
* <https://mocobeta.github.io/janome/>
* Pure Python で書かれた辞書内包の形態素解析ライブラリ
* 依存ライブラリが少ないがまだ開発途上感あり

## 係り受け解析

### CaboCha

* <https://taku910.github.io/cabocha/>
* 形態素解析部分はMeCabを使用している

## ライブラリ

### GiNZA

* <https://megagonlabs.github.io/ginza/>
* 2019/4/2公開
* 形態素解析、係り受け解析、単語依存構造解析の機能を持つ
* spaCyをフレームワークとして利用しており、SudachiPyを内部に組み込んでいる
