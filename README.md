# TOEIC Forge

英語閱讀練習的出題、審查、驗證與 HTML 輸出工具。練習涵蓋 Part 5 句子填空、Part 6 篇章填空和 Part 7 閱讀理解，附繁體中文詳解。

## 開始練習

用瀏覽器開啟 [閱讀練習總覽](examples/index.html)。HTML 可離線作答、批改、展開詳解與列印。每份 HTML 同目錄也有 `questions.json`，保留題目、選項、答案、引文、詳解和內容雜湊。

本版收錄 AI 生成的練習作品，僅從舊版交付檔中擷取最終練習內容與驗證所需的欄位。原始試題 PDF、萃取資料、OCR、舊審查紀錄和版本歷史未隨附。附帶的 JSON 使用 `document_type: practice`，驗證範圍為題型、資料格式、文內引文和內容雜湊。新出題流程保留獨立解題與審查關卡。

## 安裝與驗證

需要 Python 3.11 以上。以下指令在專案根目錄執行。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

以下指令可離線執行。

```sh
python -m toeic_engine validate examples/full-reading-08-astra/questions.json
python -m toeic_engine render examples/full-reading-08-astra/questions.json --out generated/astra-practice.html
python -m toeic_engine plan --part full --seed 42 --out generated/plan.json
python -m toeic_engine schema --out generated/unit.schema.json
```

## 生成新練習

透過環境變數設定選用的 API 金鑰，名稱見 [設定範例](.env.example)。使用 `--author-provider` 和 `--review-provider` 指定 `gemini`、`claude` 或 `openai`。下例由 Gemini 生成並審查三道句子填空。

```sh
python -m toeic_engine generate --part 5 --count 3 --out generated/new-practice
```

使用 `--part full` 可生成 100 題，`resume` 可接續生產。API 用量依題數、審查與重試次數而定，`plan` 可先查看最低呼叫數量。預設用本次及過去生成的內容做字詞重疊檢查。若另有可使用的相容參考資料，可自行透過 `--corpus` 指定路徑。

`scripts/` 收錄原專案的生成、修訂、審查和交付工具。下例會準備藍圖及 Codex 內建代理的出題任務，由代理接續執行生成及審查。

```sh
python -m scripts.diverse_native_run --run-dir generated/native-demo init --seed 42
```
