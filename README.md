# colorectal_histology網頁辨識
anaconda虛擬環境Python版本：Python 3.10.12

所有的套件版本：

Package                      Version

absl-py                      2.1.0

asttokens                    2.4.1

astunparse                   1.6.3

blinker                      1.8.2

certifi                      2024.6.2

charset-normalizer           3.3.2

click                        8.1.7

colorama                     0.4.6

comm                         0.2.2

contourpy                    1.2.1

cycler                       0.12.1

debugpy                      1.8.2

decorator                    5.1.1

dm-tree                      0.1.8

docstring_parser             0.16

etils                        1.7.0

exceptiongroup               1.2.0

executing                    2.0.1

Flask                        3.0.3

flatbuffers                  24.3.25

fonttools                    4.53.0

fsspec                       2024.6.1

gast                         0.6.0

google-pasta                 0.2.0

grpcio                       1.64.1

h5py                         3.11.0

idna                         3.7

immutabledict                4.2.0

importlib_metadata           8.0.0

importlib_resources          6.4.0

ipykernel                    6.29.5

ipython                      8.26.0

itsdangerous                 2.2.0

jedi                         0.19.1

Jinja2                       3.1.4

jupyter_client               8.6.2

jupyter_core                 5.7.2

keras                        3.4.1

kiwisolver                   1.4.5

libclang                     18.1.1

Markdown                     3.6

markdown-it-py               3.0.0

MarkupSafe                   2.1.5

matplotlib                   3.9.0

matplotlib-inline            0.1.7

mdurl                        0.1.2

mkl-fft                      1.3.8

mkl-random                   1.2.4

mkl-service                  2.4.0

ml-dtypes                    0.3.2

namex                        0.0.8

nest_asyncio                 1.6.0

numpy                        1.25.2

opt-einsum                   3.3.0

optree                       0.11.0

packaging                    24.1

pandas                       2.2.2

parso                        0.8.4

pickleshare                  0.7.5

pillow                       10.4.0

pip                          24.0

platformdirs                 4.2.2

promise                      2.3

prompt_toolkit               3.0.47

protobuf                     3.20.3

psutil                       6.0.0

pure-eval                    0.2.2

pyarrow                      16.1.0

Pygments                     2.18.0

pyparsing                    3.1.2

python-dateutil              2.9.0

pytz                         2024.1

pywin32                      306

pyzmq                        26.0.3

requests                     2.32.3

rich                         13.7.1

scipy                        1.14.0

setuptools                   69.5.1

simple_parsing               0.1.5

six                          1.16.0

stack-data                   0.6.2

tensorboard                  2.16.2

tensorboard-data-server      0.7.2

tensorflow                   2.16.2

tensorflow-datasets          4.9.6

tensorflow-intel             2.16.2

tensorflow-io-gcs-filesystem 0.31.0

tensorflow-metadata          1.15.0

termcolor                    2.4.0

toml                         0.10.2

tornado                      6.4.1

tqdm                         4.66.4

traitlets                    5.14.3

typing_extensions            4.12.2

tzdata                       2024.1

urllib3                      2.2.2

wcwidth                      0.2.13

Werkzeug                     3.0.3

wheel                        0.43.0

wrapt                        1.16.0

zipp                         3.19.2

-------------------------------------------------------------------------------------

訓練用：

TensorFlow Dataset colorectal_histology
https://www.tensorflow.org/datasets/catalog/colorectal_histology

unknown資料集
從twitter的繪師與網路上的輕小說插圖蒐集，不方便提供(請自行新增unknown資料集，共需625張圖片)

測試用：

CRC-VAL-HE-7K
https://zenodo.org/records/1214456

Roboflow資料集
https://universe.roboflow.com/vit-quoks/vit-kww1n

檔案功能

predict:辨識用

train:訓練模型(未增加unknown類別，先讓模型學習如何分類8類直腸癌類別)

train1:訓練模型(使用train所訓練的模型另外增加unknown類別的資料集訓練來防止使用者上傳非直腸癌圖片)

圖片格式轉換:將unknown資料集圖片統一轉換為png檔(非統一圖檔會讓圖片無法正常載入和訓練)

Kather_texture_2016_image_tiles_5000.zip:TensorFlow Dataset下載的colorectal_histology的資料集(使用tfds.load下載，內有5000張圖共8類，每類共625張圖)

CRC-VAL-HE-7K.zip:測試用資料集(有些類別的圖片擷取方法與TensorFlow Dataset的不同而導致辨識率不高)

網頁程式碼與使用的模型、訓練、測試資料
https://mega.nz/folder/fAASkIYD#wHIa66AXUzvOMp6t7jKwoA
