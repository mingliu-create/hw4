# MNIST 手寫數字辨識專案 (PyTorch + Streamlit)

這是一個使用 PyTorch 實作的深度學習專案，用於辨識 MNIST 手寫數字資料集。專案包含兩種模型 (MLP 和 CNN) 的訓練腳本，並附帶一個使用 Streamlit 建立的互動式網頁應用程式，讓使用者可以直接在畫布上寫數字並看到即時的辨識結果。

此專案由 AI 程式助理協助開發完成。

## 專案結構

```
project_root/
├─ app.py               # Streamlit 網頁應用程式入口
├─ train_mnist.py       # 模型訓練腳本
├─ models/
│  ├─ mlp.py            # MLP 模型定義
│  └─ cnn.py            # CNN 模型定義
├─ utils/
│  ├─ data.py           # 資料載入與預處理函式
│  └─ train_utils.py    # 共用的訓練與評估函式
├─ saved_models/        # 存放訓練好的模型權重檔
├─ requirements.txt     # Python 依賴套件列表
└─ README.md            # 本說明檔案
```

## 安裝步驟

1.  **複製專案倉庫 (Clone Repository):**
    ```bash
    git clone <您的 GitHub 倉庫網址>
    cd <專案資料夾名稱>
    ```

2.  **建立並啟用虛擬環境 (建議):**
    一個獨立的虛擬環境可以避免套件版本衝突。

    *   Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   macOS / Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **安裝所需套件:**
    ```bash
    pip install -r requirements.txt
    ```

## 如何使用

### 1. 訓練模型

您可以透過命令列來訓練 CNN 或 MLP 模型。

*   **訓練 CNN 模型 (預設 & 建議):**
    ```bash
    python train_mnist.py --model cnn
    ```
    訓練完成後，模型權重會儲存於 `saved_models/cnn_mnist.pth`。

*   **訓練 MLP 模型 (基準線):**
    ```bash
    python train_mnist.py --model mlp
    ```
    訓練完成後，模型權重會儲存於 `saved_models/mlp_mnist.pth`。

### 2. 執行 Streamlit 網頁應用程式

請先確認您已經訓練並儲存了 CNN 模型 (`cnn_mnist.pth`)，因為這是 App 預設載入的模型。

*   **啟動應用程式:**
    ```bash
    streamlit run app.py
    ```
    執行後，您的瀏覽器會自動開啟一個新分頁，顯示手寫數字辨識的互動介面。您可以在畫布上寫下數字，觀看模型的即時預測結果。
