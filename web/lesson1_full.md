# 第一堂:視覺模型基礎 — CNN, ResNet, YOLO

**時間:約 2 小時**
**目標:讓學弟們真正搞懂「電腦怎麼看圖」,從 CNN 的直覺出發,理解每一代架構在解決什麼問題,最後能自己刻一個 YOLO**

---

## 課程節奏總覽

| 段落 | 時間 | 內容 | 互動環節 |
|------|------|------|----------|
| 0. 開場 | 5 min | 為什麼要學這個 | 投票:你覺得電腦看圖跟人類有什麼不同 |
| 1. CNN 完整複習 | 30 min | Convolution / Pooling / FC,以及輔助組件 | 互動網頁:Convolution Playground |
| 2. ResNet 全變體 | 25 min | 梯度消失、Skip Connection、Basic vs Bottleneck、Pre-activation | 互動網頁:Gradient Vanishing Demo |
| 3. YOLO 系列演進 | 35 min | v1 ~ v8 完整演進,每代解決什麼問題 | 互動網頁:YOLO Anchor Visualizer |
| 4. PyTorch 實作 | 20 min | nn.Module / forward / 訓練流程 | 互動網頁:PyTorch Model Builder |
| 5. Q&A + 作業說明 | 5 min | 作業要求、評分重點 | — |

---

## 0. 開場(5 分鐘)

開場不要急著進技術,先丟一個問題給大家:

> 「如果你今天要寫一個程式,告訴電腦『這張圖裡有沒有貓』,你會怎麼寫?」

讓學弟思考 30 秒,可能的答案:
- 「找毛?」 → 那毛要怎麼定義?
- 「找耳朵?」 → 耳朵的形狀怎麼描述?
- 「比對範本?」 → 那貓的姿勢千變萬化怎麼辦?

**結論:用傳統規則寫不出來,所以才需要讓模型自己學特徵。這就是 CNN 在做的事 — 把 feature engineering 交給模型自己。**

---

## 1. CNN 完整複習(30 分鐘)

### 1.1 為什麼是 CNN,不是普通的全連接層

先講一個直覺:如果今天有一張 224×224 的彩色圖,那就是 224×224×3 = 150,528 個數字。如果直接接一個全連接層(FC)到 1024 個 neuron,參數量是:

```
150,528 × 1024 ≈ 1.5 億個參數
```

**這還只是第一層**。問題不只是參數爆炸,還有:

1. **空間關係被打散** — FC 把圖攤平成一維,左上角的像素跟右下角的像素變成「平等」的,但圖片裡相鄰的像素是有關係的
2. **位置敏感** — 貓在左上角跟貓在右下角,FC 會學成兩件事
3. **完全沒有 inductive bias** — 模型對「什麼是圖片」一無所知

CNN 的設計就是針對這三件事下手,給模型三個內建假設(這也是為什麼 CNN 在影像上比 FC 強這麼多)。

### 1.2 CNN 的三大設計特性

這三個特性是 CNN 厲害的根本原因,要記住:

| 特性 | 在做什麼 | 為什麼重要 |
|------|----------|-----------|
| **Local Connectivity** | 每個 neuron 只看前一層的小區域(receptive field) | 模仿人類視覺皮質,符合「相鄰像素相關」的事實 |
| **Weight Sharing** | 同一個 filter 滑過整張圖,參數共用 | 大幅減少參數;同一種特徵不管在哪都認得 |
| **Hierarchical Features** | 淺層學邊緣紋理,深層學部件語意 | 跟人類視覺處理方式一樣 |

### 1.3 三種核心層

#### (A) Convolutional Layer(卷積層)

用白話講:**就是拿一個小窗戶(kernel)在圖上滑,每滑一次就做一次點積。**

舉例,一個 3×3 的 kernel 在圖上滑:

```
原圖              Kernel        輸出某個位置
[1 2 3 0]        [1 0 -1]      
[4 5 6 1]        [1 0 -1]      
[7 8 9 2]        [1 0 -1]      → 1*1 + 2*0 + 3*(-1) + 4*1 + ... = ?
[0 1 2 3]
```

**重點觀念:**
- Kernel 的值是**學出來的**,不是寫死的
- 同一個 kernel 在整張圖共用 → **參數共享(parameter sharing)**
- 一個 kernel 抓一種特徵(邊緣、紋理、顏色變化)
- 一層通常會有 32、64、128、甚至 512 個 kernel,各自抓不同特徵

> 直覺類比:就像給模型一堆放大鏡,每個放大鏡專看一種東西。早期層的放大鏡看「邊」、「角」,深層的放大鏡看「眼睛」、「車輪」這種高階概念。

**重要超參數:**

| 參數 | 在做什麼 | 影響 |
|------|---------|------|
| Kernel size | 窗戶多大 | 1×1, 3×3, 5×5, 7×7 都常見;**3×3 是工業界共識** |
| Stride | 一次滑幾格 | stride=2 會讓輸出尺寸減半 |
| Padding | 邊緣補幾圈 0 | padding=1 + kernel=3 + stride=1 → 尺寸不變(`same` padding) |
| Channel / Depth | 用幾個 kernel | 越多抓越多特徵,但參數越多 |

**輸出尺寸公式**(這個一定要記):
```
Output = floor((Input + 2*Padding - Kernel) / Stride) + 1
```

**1×1 Conv 為什麼重要?**

看起來 1×1 conv 沒做空間運算,只是逐點相乘,有什麼用?
- **改變 channel 數**(降維/升維):例如把 256 channels 壓到 64 channels 再做 3×3 conv,計算量大幅降低
- **跨 channel 的線性組合**:不同特徵之間做混合
- **加非線性**(後接 ReLU):增加模型表達能力但幾乎不增加參數

這個技巧在 ResNet Bottleneck 跟所有後續架構都用到,要記住。

#### (B) Pooling Layer(池化層)

Pooling 通常用 MaxPool 2×2,做的事情:

```
[1 3 | 2 4]
[2 1 | 0 5]   →   [3 5]
[-----------]      [4 9]
[4 0 | 7 9]
[1 2 | 8 3]
```

三種 pooling:
- **Max Pooling**:取區域內最大值,**最常用**(保留最強特徵)
- **Average Pooling**:取平均,較少用於分類
- **Global Average Pooling (GAP)**:把整張 feature map 縮成一個數字,**取代 FC 層**(ResNet、現代架構都這樣做)

**為什麼要 Pooling?**
1. **降低計算量** — 尺寸減半,後面層的計算就少 4 倍
2. **增加感受野(receptive field)** — 同樣 3×3 的 kernel,在 pooling 後等於看更大範圍
3. **平移不變性(translation invariance)** — 物體稍微移動一點,輸出不會差太多

> 補充:現代很多架構(像 ResNet 後期、YOLO v3 之後、Transformer-based 模型)會用 stride=2 的 conv 取代 pooling,效果差不多但更可學習。Darknet-53 就是純全卷積,完全沒用 pooling。

#### (C) Fully Connected Layer(全連接層)

到了模型最後,通常會把特徵圖攤平接 FC,做最後的分類。

**經典 CNN 的長相(LeNet-5 / AlexNet 風格):**
```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Softmax → Output
```

但這個 pattern 在物件偵測(YOLO)裡會被改掉。**現代架構幾乎都用 GAP + 1×1 Conv 取代 FC**,因為 FC 參數太多容易 overfit。

### 1.4 常見輔助組件(現代 CNN 的標配)

光有三大核心層還不夠,實務上一定會搭配:

#### Batch Normalization(BN)

對每個 batch 的 activation 做標準化:
```
y = γ * (x - μ_batch) / σ_batch + β
```

其中 γ, β 是可學參數。BN 的好處:
- **加速收斂**(可以用更大 learning rate)
- **穩定訓練**(減少 internal covariate shift)
- **輕微 regularization 效果**(用了 BN 通常可以不用 Dropout)

> 注意:BN 在 train 跟 eval 模式行為不同。Train 時用當前 batch 的統計量,eval 時用 running average。**忘記切換 `model.train()` / `model.eval()` 是新手最常見的雷之一。**

#### Activation Function

| 名字 | 公式 | 用在哪 |
|------|------|--------|
| ReLU | max(0, x) | 最經典,YOLOv1~v3 主要用 |
| Leaky ReLU | max(0.01x, x) | 解決 dying ReLU 問題,Darknet 系列愛用 |
| Mish | x * tanh(softplus(x)) | YOLOv4 用,平滑且非單調 |
| SiLU / Swish | x * sigmoid(x) | YOLOv5+ 主要用,效果好 |

#### Dropout

訓練時隨機丟掉一部分 neuron(通常 0.5),強迫模型學 redundant features,防 overfit。**有 BN 之後,Dropout 在 CNN 裡幾乎不用了。** 但在 FC 層還是有用。

### 1.5 經典 CNN 演進史(快速帶過)

```
LeNet-5 (1998)    → CNN 開山始祖,5 層
   ↓
AlexNet (2012)    → 8 層,GPU 訓練,首次贏 ImageNet
   ↓  
VGG (2014)        → 16/19 層,全用 3×3 conv 證明深度的價值
   ↓
GoogLeNet (2014)  → 22 層,Inception module(多分支),YOLOv1 backbone 啟發
   ↓
ResNet (2015)     → 152 層,Skip Connection 解決深度問題 ★★★
   ↓
DenseNet, EfficientNet, ConvNeXt...
```

> 重點:**從 ResNet 開始,「深度不再是問題」**,後續所有架構都建立在 Skip Connection 之上(包括 YOLO 系列)。

### 互動環節 1(5 分鐘)

打開互動網頁的 **「Convolution Playground」** tab:
- 自己拖拉 kernel 的數值,看不同 kernel 對圖片有什麼影響
- 試試 Sobel kernel → 邊緣檢測
- 試試 Gaussian kernel → 模糊
- 試試 Sharpen kernel → 銳化
- 自己亂調 → 看會發生什麼

**重點要他們體會:CNN 的第一層其實就是在做這種事,只是 kernel 是學出來的。早期人類花幾十年研究的 hand-crafted feature(SIFT, HOG, Sobel...),CNN 訓練幾個 epoch 就能自己長出來。**

---

## 2. ResNet 全變體(25 分鐘)

### 2.1 為什麼需要 ResNet

ResNet 由微軟研究院的 Kaiming He(何愷明)等人在 2015 年提出,論文叫《Deep Residual Learning for Image Recognition》。**152 層的網路在 ILSVRC 2015 拿下 3.57% top-5 error**,比 VGG 深 8 倍但運算複雜度更低。

**這篇論文是 deep learning 史上引用最多的論文之一**,後續所有現代架構幾乎都用了 skip connection,包括 Transformer、YOLO 系列、Diffusion Model。

### 2.2 動機:網路越深一定越好嗎

2014 年大家發現一個怪現象:**把 CNN 疊得很深,訓練 loss 居然變高**(不是 overfitting,是 train loss 就高)。

這違反直覺,理論上深層網路至少應該跟淺層一樣好(深層的可以把多出來的層學成 identity mapping)。但實務上做不到。

```
20 層的 CNN  →  train error: 5%
56 層的 CNN  →  train error: 11%   (蛤?)
```

這個問題叫做 **degradation problem**,**不是過擬合(因為 train error 也變高)**,是優化問題 — 我們無法把那麼深的網路訓練起來。

### 2.3 兇手:梯度消失(Gradient Vanishing)

回想 backprop 的鏈式法則,梯度從輸出層往輸入層傳的時候,每經過一層就乘一次該層的偏導數。

如果每層的梯度都小於 1(例如 0.5),那:
- 10 層後:0.5^10 ≈ 0.001
- 50 層後:0.5^50 ≈ 10^(-15) → 等於 0

**結果:前面的層根本收不到梯度,等於沒在學。**

> 直覺類比:從一樓傳話到 50 樓,每傳一層走音 50%,到頂樓已經完全聽不懂。

### 2.4 ResNet 的解法:殘差學習(Residual Learning)

何愷明的核心想法:**讓網路學的不是直接的映射 H(x),而是殘差 F(x) = H(x) - x**。

```
y = F(x) + x
```

換句話說,模型只要學「我要在 x 上加什麼東西」,而不是「我要產出什麼」。當這一塊真的沒用,模型可以把 F(x) 學成 0,等於 identity mapping,不會比原本差。

**架構上長這樣:**
```
原本:  x → [Conv → BN → ReLU → Conv → BN] → ReLU → output
                    F(x)

ResNet: x → [Conv → BN → ReLU → Conv → BN] → (+) → ReLU → output
        |________________________________________↑
                 直接加過去 (skip connection)
```

**為什麼這樣有用?(三個視角)**

1. **梯度直接通路** — backward 時梯度可以直接從後面層流到前面層,不會被中間層稀釋(因為 ∂(F+x)/∂x = ∂F/∂x + 1,那個 +1 保證梯度永遠有東西)
2. **學殘差比學整體容易** — 模型只要學差量,起點已經很近
3. **退化解(degradation solution)** — 如果這一塊沒用,F(x) 學成 0 就好,不會劣化

### 2.5 兩種殘差區塊

ResNet 有兩種 building block,看模型大小決定用哪一種:

#### Basic Block(用於 ResNet-18, ResNet-34)

```
Input ──┬─→ Conv 3×3 → BN → ReLU → Conv 3×3 → BN ──┐
        │                                            (+) → ReLU → Output
        └─────────────── Identity ───────────────────┘
```

- 兩層 3×3 conv
- 速度快、計算便宜
- 適合小模型

#### Bottleneck Block(用於 ResNet-50, 101, 152)

```
Input ──┬─→ Conv 1×1 (降維) → BN → ReLU
        │   → Conv 3×3 → BN → ReLU
        │   → Conv 1×1 (升維) → BN ─────┐
        │                               (+) → ReLU → Output
        └─────────── Identity ──────────┘
```

- 三層 conv:1×1 降維 → 3×3 計算 → 1×1 升維
- **省參數**:相較於兩層 3×3 64 + 3×3 256,參數從 294,912 降到 69,632
- 更多 conv 引入更多非線性,容量更大

> Bottleneck 的精神:**讓貴的運算(3×3 conv)在比較少的 channel 上做**。這個技巧後續所有架構都沿用,YOLO 也是。

### 2.6 完整層配置(Stage 設計)

ResNet 由 5 個階段組成,所有變體都遵循這個結構:

| Stage | Output | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|-------|--------|-----------|-----------|-----------|------------|------------|
| conv1 | 112×112 | 7×7, 64, stride 2 | ← | ← | ← | ← |
| | 56×56 | 3×3 maxpool, stride 2 | ← | ← | ← | ← |
| conv2_x | 56×56 | BasicBlock × 2 | BasicBlock × 3 | Bottleneck × 3 | × 3 | × 3 |
| conv3_x | 28×28 | × 2 | × 4 | × 4 | × 4 | × 8 |
| conv4_x | 14×14 | × 2 | × 6 | × 6 | × 23 | × 36 |
| conv5_x | 7×7 | × 2 | × 3 | × 3 | × 3 | × 3 |
| | 1×1 | GAP, 1000-d FC, Softmax | ← | ← | ← | ← |
| **FLOPs** | | 1.8G | 3.6G | 3.8G | 7.6G | 11.3G |

**幾個觀察重點:**
- 每個 stage 的 channel 數翻倍(64→128→256→512),空間尺寸減半 → **資訊量大致守恆**
- 用 GAP 取代 FC,大幅減少參數量
- 整個架構非常規律,**後面所有 backbone 設計都沿用這個 pattern**(包括 Darknet-53、CSPDarknet)

### 2.7 下採樣機制(Downsampling)

在 conv3_x、conv4_x、conv5_x 的**第一個** residual block,第一個 3×3 conv 用 stride=2 來下採樣。但這時候 input 跟 output 的 channel/spatial 不匹配,沒辦法直接相加,所以 shortcut 也要對應調整 — **用 1×1 stride=2 的 conv 做 projection**(這叫 projection shortcut)。

```python
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride),
        nn.BatchNorm2d(out_channels)
    )
else:
    self.shortcut = nn.Identity()  # 直接用 x
```

### 2.8 Pre-activation ResNet(ResNet v2)

何愷明在 2016 年又發了一篇《Identity Mappings in Deep Residual Networks》,提出 **pre-activation** 設計。

把 BN, ReLU 從 conv 之後移到 conv 之前:

```
原版 (post-activation):  x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
v2 (pre-activation):    x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
```

**為什麼這樣更好?**
- 原版:shortcut 加完還要過 ReLU,**梯度路徑被破壞**(ReLU 的負半邊梯度為 0)
- v2:shortcut 是純 identity,**梯度直接無損傳遞**

結果:**可以訓練到 1000+ 層**(原版到 200 層就開始劣化)。

### 2.9 訓練細節(原論文)

- 輸入 224×224,per-pixel mean subtraction
- SGD with momentum 0.9,weight decay 1e-4
- Learning rate 從 0.1 開始,error plateau 時除以 10
- Batch size 256
- **沒有用 Dropout**(用 BN 取代)

### 互動環節 2(5 分鐘)

打開互動網頁的 **「Gradient Vanishing Demo」** tab:
- 一邊是普通 CNN,一邊是 ResNet
- 拖動「層數」slider,從 5 層拉到 60 層
- 看每一層的梯度大小視覺化(顏色從亮 → 暗代表梯度從大 → 小)
- 學弟會直接看到普通 CNN 在 30 層後梯度全黑,ResNet 還是亮的

---

## 3. YOLO 系列演進(35 分鐘)

### 3.1 物件偵測 vs 分類

先區分清楚兩件事:

| 任務 | 輸入 | 輸出 |
|------|------|------|
| 分類(Classification) | 一張圖 | 這張圖是什麼類別 |
| 偵測(Detection) | 一張圖 | 圖裡有哪些物件 + 它們在哪(bounding box)+ 各自是什麼類別 |

偵測比分類難很多,因為要同時回答 **「在哪」** 和 **「是什麼」**。

### 3.2 在 YOLO 之前:Two-stage Detector

YOLO 出現之前,主流是 R-CNN 系列(R-CNN → Fast R-CNN → Faster R-CNN),做法是:

1. **第一階段** — 用 region proposal 找出可能有物體的區域
2. **第二階段** — 對每個區域做分類

**問題:慢**。Faster R-CNN 在 2015 年大概 5~7 FPS,沒辦法即時。

### 3.3 YOLOv1(2016):One-stage 開山之作

> 論文:*You Only Look Once: Unified, Real-Time Object Detection* (Redmon et al., 2016)
> 關鍵字:**單階段、grid prediction、real-time**

#### 核心思想

把偵測當成一個 regression 問題,**一次到底**。

#### 架構(GoogLeNet 啟發版)

- **輸入**:448×448×3
- **24 層卷積 + 2 層 FC**
- 卷積層交替使用 1×1(降維)+ 3×3
- 啟發自 GoogLeNet,但用 1×1 reduction + 3×3 取代 Inception module

```
Input (448×448×3)
   ↓
24 層 Conv (1×1 + 3×3 交替)
   ↓
2 層 FC
   ↓
Output: 7×7×30 tensor
```

#### 輸出設計:S × S × (B × 5 + C)

- **S = 7**:網格大小 7×7
- **B = 2**:每個 grid cell 預測 2 個 bounding box
- **C = 20**:PASCAL VOC 20 類
- 每個 box 預測 (x, y, w, h, confidence)

**最終輸出 tensor**:7 × 7 × (2×5 + 20) = **7 × 7 × 30**

#### Loss 函數(Multi-part Sum-Squared Error)

```
Loss = λ_coord × Σ (定位誤差: x, y, w, h)
     + Σ (含物件 box 的 confidence error)
     + λ_noobj × Σ (不含物件 box 的 confidence error)
     + Σ (含物件 cell 的分類誤差)
```

- **λ_coord = 5**(強化定位的重要性)
- **λ_noobj = 0.5**(弱化大量背景的影響,因為 7×7 大部分格子都沒物體)
- 寬高用 √w, √h(解決大小框誤差不對稱問題 — 大框差 5 pixel 不嚴重,小框差 5 pixel 很慘)

#### 限制(後續版本要解決的)

- 每個 cell 只能預測一類 → **群體小物件偵測差**
- 沒有 anchor box,定位粗
- 對奇怪比例物件泛化差
- 用 FC 層,參數多

### 3.4 YOLOv2 / YOLO9000(2016/2017)

> 論文:*YOLO9000: Better, Faster, Stronger*
> 關鍵字:**Anchor box、Batch Norm、Multi-scale training**

#### Backbone:Darknet-19

- 19 層卷積 + 5 層 max pooling
- 主要用 3×3 + 1×1 卷積
- 每層後加 BN
- ImageNet 上 72.9% top-1 / 91.2% top-5

#### 七大改進(這個一定要記)

| 改進 | 內容 | 效果 |
|------|------|------|
| 1. Batch Normalization | 所有 conv 後加 BN,移除 dropout | mAP +2% |
| 2. High Resolution Classifier | 用 448×448 fine-tune 分類器 10 epoch | mAP +4% |
| 3. **Anchor Boxes** | 移除 FC,改用 anchor。輸入改 416×416 | recall ↑ |
| 4. Dimension Clusters | k-means(IoU 距離)在訓練集找 anchor | 比手選好 |
| 5. Direct Location Prediction | sigmoid 限制中心在 cell 內 | 訓練穩定 |
| 6. Fine-Grained Features | passthrough layer 接 26×26 → 13×13 | 小物件 |
| 7. Multi-Scale Training | 每 10 batches 換尺寸{320, 352, ..., 608} | 多解析度泛化 |

#### Anchor 是什麼?(超重要觀念)

想像你預先定義好幾個常見的 box 形狀(瘦高、寬扁、正方形...),每個格子上都「掛」這些 anchor,**模型只要預測「目標 box 相對於 anchor 的調整量」**,而不是直接預測絕對座標。

```
bx = σ(tx) + cx       # 中心 x = sigmoid(預測量) + 格子 x 座標
by = σ(ty) + cy       # 中心 y 同理
bw = pw × e^(tw)      # 寬度 = anchor 寬 × e^(預測量)
bh = ph × e^(th)      # 高度 = anchor 高 × e^(預測量)
```

這樣比直接預測 (x, y, w, h) 容易學很多,因為**起點已經很接近答案了**。模型只要預測「微調」,不用從零開始。

#### YOLO9000 的特殊貢獻

用 WordTree 整合 ImageNet(分類資料)+ COCO(偵測資料),可偵測 9000+ 類別,即使該類別沒有偵測標註。

### 3.5 YOLOv3(2018):Multi-scale Prediction

> 論文:*YOLOv3: An Incremental Improvement*
> 關鍵字:**Darknet-53、三尺度預測、FPN-like**

#### Backbone:Darknet-53

- **53 層 conv**,加入 ResNet 的 skip connection
- **完全沒有 pooling**,用 stride=2 conv 取代
- top-1 / top-5 與 ResNet-152 相當,**速度快兩倍**

```
Layer Type     Filters   Size     Output
Conv           32        3×3      256×256
Conv           64        3×3/2    128×128
[1× Residual]
  Conv         32        1×1
  Conv         64        3×3
Conv           128       3×3/2    64×64
[2× Residual]
Conv           256       3×3/2    32×32
[8× Residual]      ← 輸出至 detection scale 3 (52×52)
Conv           512       3×3/2    16×16
[8× Residual]      ← 輸出至 detection scale 2 (26×26)
Conv           1024      3×3/2    8×8
[4× Residual]      ← 輸出至 detection scale 1 (13×13)
```

#### 多尺度預測(類 FPN)

YOLOv3 在**三個不同尺度**的 feature map 上預測,這個設計影響了後面所有 YOLO 版本:

```
Input 416×416
  ↓
Darknet-53
  ↓
  ├──→ 13×13 feature map  → 預測大物體
  ├──→ 26×26 feature map  → 預測中物體   (上採樣後 concat 淺層)
  └──→ 52×52 feature map  → 預測小物體   (再上採樣後 concat 更淺層)
```

#### Anchor 設定

每個尺度用 3 個 anchor,共 9 個(k-means 在 COCO 上得到):
- **13×13(大物體)**:(116×90), (156×198), (373×326)
- **26×26(中物體)**:(30×61), (62×45), (59×119)
- **52×52(小物體)**:(10×13), (16×30), (33×23)

#### 輸出張量

每個 scale 輸出 `S × S × [3 × (4 + 1 + 80)]` = `S × S × 255`(COCO 80 類)

#### 分類改變:Multi-label

YOLOv3 用 **sigmoid + binary cross-entropy 取代 softmax**,因為:
- 一個物體可能屬於多個類別(例如 Woman + Person 都對)
- COCO 的 label 不是互斥的
- BCE 訓練更穩定

### 3.6 YOLOv4(2020):工程集大成

> 論文:*YOLOv4: Optimal Speed and Accuracy of Object Detection* (Bochkovskiy, Wang, Liao)
> 關鍵字:**Backbone+Neck+Head、Bag of Freebies、Bag of Specials**

#### 三段式架構(YOLO 系列首次明確採用)

```
Input → Backbone (CSPDarknet53) → Neck (SPP + PANet) → Head (YOLOv3 head)
```

**從 v4 開始,所有 YOLO 都長這樣**,不只 YOLO,幾乎所有現代 detector 都是這個結構。

#### Backbone:CSPDarknet-53

在 Darknet-53 加入 **CSPNet (Cross Stage Partial Network)**:
- 把 base layer 的 feature map **切成兩半**
- 一半進 dense block,一半 shortcut
- 最後 concat

效果:減少計算、提升梯度流,**梯度不會在重複的 dense block 中重複計算**。

激活函數從 Leaky ReLU 換成 **Mish**:`x * tanh(softplus(x))`,平滑且非單調。

#### Neck:SPP + PANet

**SPP (Spatial Pyramid Pooling)**:多尺度 max pool(5×5, 9×9, 13×13)concat,擴大 receptive field 但不增加計算量。

**PANet (Path Aggregation Network)**:FPN 是 top-down(深層特徵往上傳到淺層),PANet 多加一條 **bottom-up**(淺層特徵也傳到深層),雙向強化。

```
        Top-down (FPN)        Bottom-up (PAN)
P5 ──→ P5'              P5'' ←── (從 P4'' 來)
       ↓                  ↑
P4 ──→ P4'              P4'' ←── (從 P3'' 來)
       ↓                  ↑
P3 ──→ P3'              P3''
                          ↑
                        (P3' 直接往上)
```

#### Head:沿用 YOLOv3 head(anchor-based, 3 scales)

#### Bag of Freebies (BoF) — 不增加推論成本

只在訓練時用,推論不影響速度:
- **Mosaic data augmentation**(4 張圖拼一張,讓模型一次看到很多 context)
- **CutMix**
- **Class label smoothing**(讓 one-hot 變成 0.9/0.1,避免過度自信)
- **DropBlock regularization**(整塊整塊地 dropout)
- **CIoU Loss** 取代 MSE — 同時考慮 overlap、中心距離、長寬比

#### Bag of Specials (BoS) — 略增成本但提精度

- Mish activation
- SPP
- PAN
- DIoU-NMS(NMS 時考慮中心距離)

#### 性能

COCO test-dev 2017:**AP 43.5%, AP50 65.7%**,V100 GPU **超過 50 FPS**。

### 3.7 YOLOv5(2020, Ultralytics):工業界主流

> 不是學術論文,**但因為好用紅了**

#### 主要貢獻(都是工程上的)

- **PyTorch 實作**(前面都是 Darknet C 語言)
- 訓練流程超簡單(`yolo train` 一行)
- 提供 5 種尺寸:n / s / m / l / x
- 部署友善(ONNX, TensorRT, CoreML 都好轉)
- 配置檔從 `.cfg` 改成 `.yaml`,可讀性高

#### 架構

```
Input (640×640)
   ↓
[Backbone: CSPDarknet53 改良版]
   - Focus(後期改成 6×6 Conv stem)
   - C3 module (CSP Bottleneck × N)
   ↓
[Neck: SPPF + CSP-PAN]
   ↓
[Head: 3-scale YOLOv3-style head]
   - 80×80 (small), 40×40 (medium), 20×20 (large)
```

#### 關鍵組件

**Focus 層(早期版本)**:把輸入做 slice 重組,把空間資訊塞進 channel 維度。後期版本被替換成 6×6 Conv2d,效率更好。

**C3 Module**:CSP 結構簡化版,3 個 1×1 conv + N 個 bottleneck。

**SPPF (Spatial Pyramid Pooling - Fast)**:用 3 個串聯的 5×5 max pool 取代 SPP 的並聯多尺寸 pooling,**結果等價但更快**。

**CSP-PAN**:把 CSP 結構併入 PAN,提升訓練效率與精度。

#### 模型尺寸

| 模型 | depth_multiple | width_multiple | 參數量 | 用途 |
|------|----------------|----------------|--------|------|
| YOLOv5n | 0.33 | 0.25 | ~1.9M | nano,邊緣裝置 |
| YOLOv5s | 0.33 | 0.50 | ~7.2M | small |
| YOLOv5m | 0.67 | 0.75 | ~21.2M | medium |
| YOLOv5l | 1.00 | 1.00 | ~46.5M | large |
| YOLOv5x | 1.33 | 1.25 | ~86.7M | extra large |

#### 訓練細節

- **Anchor-based**(仍保留),3 scales × 3 anchors
- **Mosaic + MixUp** 增強
- **Auto-anchor**(自適應 anchor 計算 — 訓練前自動 k-means)
- **Letterbox**(自適應圖片縮放,保持 aspect ratio)
- Loss:BCE (cls, obj) + CIoU (box)

### 3.8 YOLOv6(2022, 美團)

> 主打**工業部署**,backbone-neck-head 模組化

#### Backbone:EfficientRep

關鍵是 **RepVGG block**:
- **訓練時**:用多分支結構(3×3 + 1×1 + identity),**好訓練**
- **推論時**:用「結構重參數化」把多分支合併成單一 3×3 conv,**極快**

這是業界很重要的技巧,叫做 **structural re-parameterization**。

#### Neck:Rep-PAN

PAN 結構,但用 RepVGG-style block。

#### Head:Efficient Decoupled Head

- **Decoupled Head**:分類跟回歸**分開預測**,避免兩個任務互相干擾(YOLOX 提出的觀念)
- **Anchor-free**:第一個拋棄 anchor 的 YOLO 系列
- 比 YOLOX 解耦頭更輕量(共享部分卷積)

#### 訓練策略

- **Task Alignment Learning (TAL)** — 比 SimOTA 更聰明的標籤分配
- Loss:
  - Classification:**VariFocal Loss (VFL)**
  - Box:**SIoU** 或 **GIoU Loss**
  - Box quality:**Distribution Focal Loss (DFL)**

#### 量化部署

- 訓練後量化(PTQ)+ 量化感知訓練(QAT)
- T4 上 YOLOv6-N 達 **1234 FPS**,AP 35.9%

### 3.9 YOLOv7(2022)

> 論文:*YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors*
> 關鍵字:**E-ELAN、planned re-param、auxiliary head**

#### 核心:從零訓練,只用 COCO

- 不在 ImageNet 預訓練
- 主要分 YOLOv7 與 YOLOv7-X
- 輸入 640×640,有 scaled 版本適配高解析度

#### Backbone:E-ELAN

**Extended Efficient Layer Aggregation Network**:
- 透過 **expand → shuffle → merge** 的 cardinality 操作
- 讓網路在不破壞原始 gradient path 下持續學習
- 控制最短/最長 gradient path

#### Trainable Bag-of-Freebies(可訓練的免費餐)

1. **Planned Re-parameterized Convolution**:避免 RepConv 的 identity 連接破壞 ResNet/DenseNet 的 gradient
2. **Coarse-to-fine Lead Head Guided Label Assignment**:
   - **Lead head**:產生最終預測
   - **Auxiliary head**:訓練時輔助
   - 用 lead head 的預測指導兩個 head 的標籤分配
3. **Batch Normalization in conv-bn-activation**:推論時融合 BN 進 conv

#### 變體

- YOLOv7-tiny(edge)
- YOLOv7(standard)
- YOLOv7-W6/E6/D6/E6E(高解析度 1280)

#### 性能

COCO 達 **56.8% AP**,5–160 FPS 範圍內 SOTA。

### 3.10 YOLOv8(2023, Ultralytics)

> 整合多任務支援(detection / segmentation / pose / classification)

#### 架構總覽

```
Input → Stem (Conv layers)
      → C2f blocks (with residuals) [Backbone]
      → SPPF [neck 起點]
      → 上採樣 + concat + C2f (no residuals) [Neck PAN]
      → Decoupled Multi-scale Detection Head
```

#### Backbone 改變:C2f 取代 C3

```
C3 (YOLOv5):                      C2f (YOLOv8):
Input → 1×1 Conv ──┐              Input → 1×1 Conv → split
       → Bottleneck × N           ├──→ Bottleneck → ─┐
       → 1×1 Conv                 ├──→ Bottleneck → ─┤
       → concat → 1×1 Conv        │   (所有輸出都 concat)
                                  └──→ 1×1 Conv → Output
```

**差別**:C3 只取最後一個 bottleneck 的輸出,C2f 把**所有 bottleneck 的輸出都 concat**,有效感受野更大、上下文更豐富。

Block 數量從 3-6-9-3 改為 3-6-6-3。

#### Neck

仍是 PAN-FPN,但**移除 1×1 降維 conv**,直接 concat 後送進 C2f。

#### Head:Decoupled + Anchor-Free

```
Feature → ┬→ Cls Branch (Conv → Conv → Conv 1×1) → 類別
          └→ Reg Branch (Conv → Conv → Conv 1×1) → DFL → bbox
```

特性:
- **Anchor-free**:直接預測物件中心 + 寬高,不再用 anchor priors
- **Decoupled**:分類、回歸獨立分支
- **DFL (Distribution Focal Loss)**:把 bbox 邊界當分布預測,而不是單一值

#### 訓練策略

- **Task Aligned Assigner (TAL)** 標籤分配
- **Mosaic augmentation**,最後 10 epoch 關閉(讓模型適應真實分布)
- Loss:
  - Classification:**BCE**
  - Box:**CIoU + DFL**
- 多任務:detection / segmentation (YOLACT-based) / pose / classification / tracking

#### 模型尺寸與性能

| 模型 | 參數量 | mAP COCO | 用途 |
|------|--------|----------|------|
| YOLOv8n | 3.2M | 37.3 | nano |
| YOLOv8s | 11.2M | 44.9 | small |
| YOLOv8m | 25.9M | 50.2 | medium |
| YOLOv8l | 43.7M | 52.9 | large |
| YOLOv8x | 68.2M | 53.9 | extra large |

### 3.11 YOLO 各版本對照總表(★ 必看)

| 版本 | 年份 | Backbone | Neck | Head | Anchor | Loss 重點 |
|------|------|----------|------|------|--------|-----------|
| **v1** | 2016 | GoogLeNet-like (24 conv) | — | FC, 7×7×30 | 無 | SSE multi-part |
| **v2** | 2016 | Darknet-19 | passthrough | conv, 13×13×125 | 5 anchors (k-means) | SSE + sigmoid 中心 |
| **v3** | 2018 | Darknet-53 | FPN-like (3 scales) | conv, multi-scale | 9 anchors (3×3) | BCE + SSE |
| **v4** | 2020 | CSPDarknet-53 + Mish | SPP + PANet | YOLOv3 head | 9 anchors | CIoU + BCE |
| **v5** | 2020 | CSPDarknet (C3) | SPPF + CSP-PAN | YOLOv3 head | 9 anchors (auto) | CIoU + BCE |
| **v6** | 2022 | EfficientRep (RepVGG) | Rep-PAN | Decoupled Anchor-free | 無 | VFL + SIoU + DFL |
| **v7** | 2022 | E-ELAN | SPPCSPC + PANet | Lead+Aux head | Anchor-based | CIoU + BCE |
| **v8** | 2023 | CSPDarknet (C2f) | SPPF + PAN (no 1×1) | Decoupled Anchor-free | 無 | CIoU + DFL + BCE |

### 3.12 通用設計脈絡演進(讀懂這段就懂 YOLO 史)

1. **Backbone 演進**:GoogLeNet-like → Darknet-19 → Darknet-53 → CSPDarknet-53 → EfficientRep / E-ELAN → C2f-CSPDarknet
2. **Neck 演進**:無 → passthrough → FPN → SPP+PAN → SPPF+CSP-PAN → Rep-PAN
3. **Head 演進**:FC → Anchor-based coupled → Decoupled Anchor-free
4. **Anchor 演進**:無 → k-means 5 個 → k-means 9 個(3 scales)→ **拋棄回到 anchor-free**
5. **Loss 演進**:SSE → BCE+SSE → CIoU+BCE → VFL+SIoU+DFL → CIoU+DFL+BCE
6. **Activation 演進**:Leaky ReLU → Mish → SiLU/Swish

> **這部分的目標不是讓他們背版本差異,是讓他們理解「每一代都在解決前一代的某個具體問題」。** 看懂演進邏輯,才知道未來會往哪走。

### 互動環節 3(5 分鐘)

打開互動網頁的 **「YOLO Anchor Visualizer」** tab:
- 拖拉 grid 大小(從 5×5 到 19×19)
- 切換 anchor-based / anchor-free 模式
- 看 anchor 怎麼覆蓋整張圖
- 點某個格子,看它要負責預測哪些 box

---

## 4. PyTorch 組模型實作(20 分鐘)

### 4.1 nn.Module 是什麼

PyTorch 裡所有的模型都繼承自 `nn.Module`。它做的事情:
1. **管理所有 parameter**(自動追蹤,給 optimizer 用)
2. **支援 GPU 移動**(`.cuda()` 一行搞定)
3. **支援 train/eval 模式切換**(影響 BN, Dropout 行為)
4. **自動 backward**(只要 forward 寫對,backward 不用自己寫)

### 4.2 一個最簡單的 CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 在 __init__ 裡定義「這個模型有哪些零件」
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 8 * 8, num_classes)  # 假設輸入 32×32
    
    def forward(self, x):
        # 在 forward 裡定義「這些零件怎麼組起來」
        x = self.relu(self.conv1(x))   # (B, 32, 32, 32)
        x = self.pool(x)                # (B, 32, 16, 16)
        x = self.relu(self.conv2(x))   # (B, 64, 16, 16)
        x = self.pool(x)                # (B, 64, 8, 8)
        x = x.view(x.size(0), -1)      # 攤平 (B, 4096)
        x = self.fc(x)                  # (B, 10)
        return x
```

**重點觀念:**
- `__init__` 定義「有什麼」
- `forward` 定義「怎麼用」
- backward 完全自動,**不用自己寫**

### 4.3 寫一個 ResNet Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果 input/output 的 shape 不一樣,需要一個 shortcut 來調整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity   # ← 這就是 skip connection
        out = self.relu(out)
        return out
```

**這幾行就是 ResNet 的精髓**。理解這個 block,就理解 ResNet 90% 的內容。

### 4.4 訓練流程怎麼跑

```python
import torch.optim as optim

# 1. 模型、loss、optimizer
model = SimpleCNN(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 2. 訓練 loop
for epoch in range(num_epochs):
    model.train()  # 切到訓練模式
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()   # 清掉上一次的梯度!忘記寫會爆
        loss.backward()         # 自動算梯度
        optimizer.step()        # 更新參數
    
    # 驗證
    model.eval()  # 切到 eval 模式(BN 用 running stats, Dropout 關掉)
    with torch.no_grad():       # 不算梯度,省記憶體
        for images, labels in val_loader:
            ...
```

**新手最常踩的雷:**
1. 忘記 `optimizer.zero_grad()` → 梯度會累加
2. 忘記 `model.train()` / `model.eval()` 切換 → BN 行為不對
3. 忘記 `with torch.no_grad():` 在驗證 → 記憶體爆掉
4. tensor 沒搬 GPU → CPU/GPU mismatch error
5. 用 `view` 而不是 `reshape` 在非 contiguous tensor 上 → runtime error

### 4.5 YOLO 跟分類模型的差別在哪

如果學弟要刻 YOLO,跟前面的 SimpleCNN 差在:

1. **Output 不是一個類別,是 (B, S, S, anchors × (5 + classes))**
   - S 是 grid size
   - 5 = (x, y, w, h, confidence)
   - classes = 你的類別數

2. **Loss 是組合 loss**:
   - **Localization loss**(box 的 x, y, w, h) — 用 MSE 或 CIoU
   - **Confidence loss**(這格有沒有物體) — 用 BCE
   - **Classification loss**(是什麼類別) — 用 BCE 或 CrossEntropy

3. **資料前處理要產 ground truth tensor**:
   - 把 annotation(每個物件的 box + class)轉成跟 output 一樣 shape 的 tensor
   - 決定哪個 cell、哪個 anchor 負責哪個 GT
   - 這是 YOLO 最複雜的部分

4. **Inference 後處理**:
   - 把 raw output 轉回 box 座標
   - **NMS (Non-Maximum Suppression)** 去掉重複的 box

這部分留給作業讓他們自己摸索,但實作網頁裡會給一個 YOLO output 結構的視覺化。

### 互動環節 4(5 分鐘)

打開互動網頁的 **「PyTorch Model Builder」** tab:
- 拖拉積木式組合層(Conv, Pool, ReLU, BN, FC)
- 即時顯示對應的 PyTorch code
- 顯示每層的輸出 shape(讓他們直觀感受 shape 怎麼變化)
- 算出總參數量

---

## 5. Q&A + 作業說明(5 分鐘)

### 作業 1 詳細說明

**題目:實作並訓練一個自己設計的 YOLO 模型**

**要求:**

1. **自己設計模型架構**
   - 可以參考 YOLOv1 ~ v8 的設計,但**不能完全照抄**
   - 至少要有自己的 backbone 設計(可以基於 ResNet/Darknet 改)
   - 可以選 anchor-based 或 anchor-free
   - **要有自己的 design choice**,不能寫「我抄 v5 的」

2. **用 PyTorch 從頭刻 nn.Module**
   - 不能用 ultralytics 的現成 YOLO
   - 不能用 `torch.hub.load`
   - backbone 可以用 torchvision.models 預訓練的(但要說明怎麼接的)

3. **寫一份簡短說明(Markdown 即可)**,要包含:
   - 模型架構圖(可以用 mermaid 或手畫拍照)
   - 每層的輸入/輸出 shape
   - **為什麼這樣設計** — 這是評分重點
   - 你考慮了哪些 trade-off
   - 為什麼選 anchor-based / anchor-free
   - Loss 怎麼設計

4. **在自選資料集上訓練到收斂**
   - 可以用 COCO subset(只取 5 個類別之類的)
   - 可以用 PASCAL VOC
   - 可以自製(例如自己拍照標 100 張)
   - **要附訓練 loss 曲線**

**評分重點(按權重排):**

| 項目 | 權重 | 說明 |
|------|------|------|
| 設計思路說明 | 40% | 你有沒有想清楚為什麼這樣做 |
| 架構合理性 | 30% | 設計有沒有明顯錯誤、合不合理 |
| 訓練完成度 | 20% | 有沒有真的訓練到收斂 |
| 準確率 | 10% | 不重要,有就好 |

> **再強調一次:重點不是準確率多高,是有沒有想清楚自己在做什麼。**
> 如果你寫「我這層用 64 channels 因為...」,我要看到「因為」後面的東西,不是「因為大家都這樣」。

**繳交方式:**

GitHub repo,包含:
- `model.py` — 模型定義
- `train.py` — 訓練 script
- `README.md` — 設計說明
- `loss_curve.png` — 訓練曲線
- 一個 inference demo(吃一張圖,輸出畫框結果)

**截止:下週上課前**

**可以問的問題:**
- 卡在 loss 設計 → 來問
- 卡在資料前處理 → 來問
- 不確定設計合不合理 → 來問
- 找不到資料集 → 來問

**不可以做的事:**
- 直接複製 GitHub 上的 YOLO 實作
- 用 ChatGPT/Claude 全部生成完跑一跑

### 課後資源

**論文必讀:**
- ResNet: https://arxiv.org/abs/1512.03385
- ResNet v2 (Pre-activation): https://arxiv.org/abs/1603.05027
- YOLOv1: https://arxiv.org/abs/1506.02640
- YOLOv3: https://arxiv.org/abs/1804.02767
- YOLOv4: https://arxiv.org/abs/2004.10934
- YOLOv7: https://arxiv.org/abs/2207.02696

**推薦影片:**
- 李宏毅 - CNN
- Aladdin Persson 的 YOLOv1 from scratch
- Yannic Kilcher 的 ResNet 論文解讀

**官方文件:**
- PyTorch 官方教學:https://pytorch.org/tutorials/
- Ultralytics YOLOv5 文件:https://docs.ultralytics.com
- Papers with Code - Object Detection:https://paperswithcode.com/task/object-detection
