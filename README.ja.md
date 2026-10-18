<p align="center"><img src="assets/banner.jpg" alt="Blodsom-1" width="100%"></p>

<p align="center"><b>TRELLIS.2 の画像→3D を VRAM 4GB で。重みも結果も元のまま。</b><br>
<a href="README.md">English</a></p>

Blodsom-1 は、Microsoft の [TRELLIS.2](https://github.com/microsoft/TRELLIS.2)(4B、画像1枚から3Dモデル)を
小さなマシンで動かすためのものです。元の TRELLIS.2 は 24GB 以上の NVIDIA GPU を求めますが、Blodsom-1 は
標準設定(1024 二段)を **VRAM 4GB の RTX 3050 Laptop と RAM 6.5GB** で完走します。Apple Silicon の Mac でも約 6GB で動きます。

モデル自体には手を入れていません。量子化・枝刈り・ステップ削減はなし。同じ画像と同じシード値なら、
潜在表現もメッシュも、同じ機器で動かした元のパイプラインと **1ビットも違わず一致**します。

## ⚠️ 遅いです

Blodsom-1 は、時間と引き換えにメモリを減らしています。今まで動かなかった機械で **とにかく動かす** ためのもので、
速く動かすためのものではありません。

| 機械 | 1024 二段、画像1枚 |
|---|---|
| NVIDIA H100(元の TRELLIS.2、公式 README の 1024³) | 約 **17 秒** |
| Apple M5 32GB(手を加えない元。trellis-mac そのまま、Metal 拡張なし、全モデルを最初に読み込み) | 約 **330 秒** |
| RTX 3050 Laptop 4GB + RAM 6.5GB(Blodsom-1) | 約 **174 秒**(3分) |
| Apple M5 32GB(Blodsom-1) | 約 **165〜180 秒** |

- **1体あたり3分前後** かかると思ってください。遅い GPU ではさらにかかります。
- 実行のたびに、各モデルをディスクから読み直します(RTX 3050 で約 10 秒)。
- `--stream`(VRAM 4GB 未満向け)はさらに遅く、以前の計測では RTX 3050 で約 300 秒でした。
- メモリに余裕があれば `--resident` で少し速くなります。M5 で 1枚目 178 秒 → 157 秒、2枚目以降 138 秒(結果は同じ、使用は最大 24GB)。大きなモデルではほぼ変わりません(330万頂点で 363 秒 → 362 秒)。
- 速さ優先のモード(ステップ削減・蒸留。結果は少し変わる)は、別オプションとして予定しています。

## どうやって収めたか

| | 元の TRELLIS.2 | Blodsom-1 |
|---|---|---|
| メモリ上のモデル | 全部(生成モデルだけで 13GB) | 1つずつ。GPU へ直接読み込み、使い終わったら捨てる |
| モデルの読み込み | いったん CPU 上に float32 で全体を作る | 空のモデルを作り、保存された値を1つずつ元の型のまま GPU へ |
| 疎な畳み込み(PyTorch のみ版) | 近傍を Python のループで探し、int64 の表と数GBの添字を作る | 並べ替えたキーと `searchsorted`、int32、`index_add_`。足し算の順番まで同じ |
| デコーダーの MLP | 全ボクセルを一度に | 262,144 ボクセルずつ |
| メッシュ化(PyTorch のみ版) | Python の辞書を数百万回引く | 並べ替えたキーの配列で一括 |
| 計算中 | — | 次のモデルのファイルを先に RAM へ読んでおく(NVIDIA) |
| 画像の特徴(DINOv3) | 同意が必要な Hugging Face のダウンロード | timm が再配布している同じ重み(登録不要) |

RTX 3050 Laptop(4GB)、1024 二段での実測:

| 使用中のモデル | 必要な VRAM | 時間 |
|---|---|---|
| 大まかな形の生成 | 2.69 GB | 44 秒 |
| 形 512 → デコーダー → 形 1024 | 2.73 GB | 70 秒 |
| 色 1024 → デコーダー | 2.81 GB | 37 秒 |
| **全体** | **GPU 全体で 4.0 GB** | **174 秒** |

RAM の使用は最大 6.1GB。6.5GB に強制的に制限しても 173.5 秒で完走し、メッシュは一致しました。

## インストール

Git と Python 3.10 以上が必要です。モデルの重み(約 15GB)は初回実行時にダウンロードされます。

**macOS(Apple Silicon)/ Linux(NVIDIA)**

```bash
git clone https://github.com/Methurcoder/Blodsom-1.git
cd Blodsom-1
bash setup.sh
```

**Windows(NVIDIA)**

```powershell
git clone https://github.com/Methurcoder/Blodsom-1.git
cd Blodsom-1
powershell -ExecutionPolicy Bypass -File setup.ps1
```

setup は [trellis-mac](https://github.com/shivampkumar/trellis-mac) と TRELLIS.2 を検証済みのコミットで
`third_party/` に取得し、trellis-mac の修正(PyTorch だけで動く注意機構・畳み込み・メッシュ化)を当て、
依存パッケージを `.venv` に入れます。コンパイラも CUDA Toolkit も要りません。

## 使い方

```bash
.venv/bin/python generate.py character.png              # macOS / Linux
.venv\Scripts\python generate.py character.png          # Windows
```

`character.glb` ができます。複数の画像をまとめて渡せます(`generate.py a.png b.png c.png`)。オプション:

| オプション | |
|---|---|
| `--type 512 / 1024 / 1024_cascade / 1536_cascade` | パイプラインの種類(既定は元と同じ `1024_cascade`) |
| `--seed N` | シード値(既定は元と同じ 42) |
| `--faces N` | GLB を約 N 面に減らす(既定は減らさない。頂点は約 200 万) |
| `--stream` | VRAM 4GB 未満向け。変換ブロックを1つずつ GPU へ送る(かなり遅い) |
| `--no-prefetch` | RAM を減らす(少し遅くなる) |
| `--ram-limit GB` | Windows:物理メモリを制限して、小さい機械を試す |
| `--save-mesh` | 元のメッシュとボクセルの値も保存(`.pt`) |
| `--resident` | メモリに余裕がある機械向け(約 24GB)。全モデルを読み込んだままにし、メモリの片付けを省く。結果は同じ |

**制限**

- 画像は **背景が透過している PNG** にしてください。元の TRELLIS.2 は RMBG-2.0 で背景を消しますが、
  これは同意が必要で非商用なので、Blodsom-1 には入れていません。透過画像なら元もこの処理を飛ばすので、結果は一致します。
- GLB の色は **頂点カラー** です(焼き込んだ PBR テクスチャではありません)。テクスチャの焼き込みには CUDA / Metal の
  拡張のビルドが必要なためです。形と各ボクセルの色は元と同じです。

## 「同じ結果」の確かめ方

`tools/equivalence.py` は、元のパイプライン(全モデルを読み込み、元の読み込み処理)と Blodsom-1 を同じ画像・シード値で動かし、
重みの全テンソル・潜在表現・メッシュを1ビット単位で比べます。`tools/measure_cuda.py` は上のモデルごとのメモリ表を出します。

「1ビットも違わない」は、同じ機器・同じ PyTorch での話です。GPU が違えば(Mac と NVIDIA でも)丸め方が変わるのは、元の TRELLIS.2 でも同じです。

検証環境:Apple M5(macOS、PyTorch 2.14)、RTX 3050 Laptop 4GB(Windows、PyTorch 2.11 + CUDA 12.8)。Linux は未検証です。`setup.ps1` はまだまっさらな Windows では試していません(上の RTX 3050 の結果は、同じパッケージを手で入れた環境です)。

## ライセンス

- Blodsom-1 自身のコード:MIT([LICENSE](LICENSE))
- TRELLIS.2 のコードと重み:MIT(Microsoft)。trellis-mac:MIT。DINOv3 の重み(実行時に timm から取得):
  [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)。
  詳しくは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Blodsom-1 は Microsoft・Meta とは関係ありません。
