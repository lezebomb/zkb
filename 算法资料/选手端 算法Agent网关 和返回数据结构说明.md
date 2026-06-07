# 选手端 · 算法 Agent 网关与返回数据结构说明（最终版）

> **规范基准**：本文与仓库内参考实现 `**初赛竞赛系统/选手算法服务-test/app.py`** 的 **请求 / 响应 JSON 字段与形状一致**；若与竞赛平台其它校验细节有出入，以 `app.py` 为准补文档，平台侧以 `ClientEvaluationResultApplier`、`VisualScoringService` 代码为准。  
> **链路**：竞赛平台（HTTPS）→ 选手浏览器 → **本机 Agent 网关**（队长在平台填写的 Base URL）→ 可转发至本队算法进程 → 响应 JSON 回传平台 → 判分落库。  
> **开放题型（与 `app.py` 中 `_SUPPORTED_TASKS` 顺序一致）**：`classify`、`ocr`、`detect`。`GET /health` 的 `**supported_tasks` 必须列出本队实际支持的子集**（须含参赛所需类型）。

---

## 一、Agent 网关部署位置（固定）


| 项             | 约定                                                                                                       |
| ------------- | -------------------------------------------------------------------------------------------------------- |
| 部署位置          | **选手本机或可被本机浏览器访问的同一内网地址**（与参赛端 `fetch` 同源策略、CORS 策略一致）                                                   |
| 登记地址          | 队长在平台填写的 **Agent Base URL** 必须为**根地址**，例如 `http://127.0.0.1:8080`（**不要**省略协议；**不要**在末尾多写 `/infer`，由平台拼接） |
| 必须实现的 HTTP 路径 | `**GET /health`**、`**POST /infer`**（路径名固定，大小写敏感）                                                         |
| 实现形态          | **二选一**：① 网关与算法同一进程实现两接口；② 网关为反向代理，将上述两路径**原样转发**至本机算法服务且**响应体不改写**                                      |


---

## 二、HTTP 与传输（固定）


| 项                  | 约定                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------- |
| 字符编码               | **UTF-8**                                                                                     |
| `POST /infer` 请求头  | `**Content-Type: application/json`**（必须）                                                      |
| `POST /infer` 成功响应 | **HTTP 状态码必须为 2xx**；响应体为 **合法 JSON**                                                          |
| `POST /infer` 协议失败 | HTTP **≥400** 或 非 JSON 正文 → **当次抽样失败**，整题按平台规则失败                                              |
| 超时                 | 单次推理须在请求 `**meta.infer_T_max_ms`**（毫秒）内完成；浏览器会在此基础上增加少量余量后中断                                  |
| 跨域                 | 竞赛页域名访问本机网关时，**必须**对 `/health`、`/infer`（及浏览器可能发起的 `**OPTIONS` 预检**）返回合规 **CORS** 头，否则浏览器无法发请求 |
| 鉴权                 | 若队长在平台配置了自定义 Header，浏览器会**自动附带**；网关**必须**按本队约定校验，失败返回 **401** 及 JSON `message`                |


---

## 三、`GET /health`（固定）

### 3.1 请求

```http
GET {agent_base_url}/health
```

`{agent_base_url}` 为队长登记的 Base URL（去掉末尾 `/` 后由平台拼接 `/health`，若登记项已以 `/health` 结尾则按参赛端逻辑不再重复追加）。

### 3.2 响应（固定要求）

- **HTTP 状态码**：**200**  
- **响应体**：**JSON 对象**  
- **字段**：


| 字段                | 类型       | 必须    | 说明                                                                                                                               |
| ----------------- | -------- | ----- | -------------------------------------------------------------------------------------------------------------------------------- |
| `status`          | string   | **是** | 平台用于判断在线；**必须为** `"ok"`（小写）表示可接受评测                                                                                               |
| `supported_tasks` | string[] | **是** | **必须**列出本机已实现的 `task_type`；须为 `classify`、`ocr`、`detect` 的子集，且**须包含本队实际要打的题型**（**顺序与 `app.py` 一致**：`classify` → `ocr` → `detect`） |
| `service`         | string   | 否     | `**app.py` 固定返回**：服务名 `contestant-algo-test-service`                                                                             |
| `version`         | string   | 否     | `**app.py` 固定返回**：与 `APP_VERSION` 一致                                                                                             |
| `bridge_mode`     | string   | 否     | `**app.py` 固定返回**：`mock-local`                                                                                                   |


### 3.3 响应示例（与 `app.py` 中 `health()` 返回一致）

```json
{
  "status": "ok",
  "supported_tasks": ["classify", "ocr", "detect"],
  "service": "contestant-algo-test-service",
  "version": "0.2.0",
  "bridge_mode": "mock-local"
}
```

自研网关可省略 `service` / `version` / `bridge_mode`；**与 `app.py` 联调对齐时以上字段以示例为准**。

---

## 四、`POST /infer` — 通用请求 / 响应信封（固定）

### 4.1 请求体 — 所有题型共用的顶层字段（必须出现）


| 字段           | 类型     | 必须                      | 说明                                                                                                                  |
| ------------ | ------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `request_id` | string | **是**                   | 平台生成，**全局唯一**；响应中**必须原样回显**                                                                                         |
| `session_id` | string | **是**                   | 平台生成；**请求必填**。响应是否回传见 §4.2（`**app.py` 不回传**）                                                                        |
| `task_type`  | string | **是**                   | **语义上**仅支持 `ocr`、`classify`、`detect`；`**app.py` 会对请求值 `strip().lower()`**，响应中的 `task_type` 一律为小写                    |
| `image`      | object | **是**                   | **必须**同时包含 `format`（string）与 `data`（string）；常见为 `format: "url"` 且 `data` 为题图 URL（`app.py` 中 `format` 缺省为 `"url"`）   |
| `meta`       | object | 否（`**app.py` 默认 `{}`**） | 判分上下文；平台通常会带 `difficulty`、`sample_index`、`sample_k`、`infer_T_max_ms` 等；**请求体省略 `meta` 时 `app.py` 视为空对象**；题型相关字段见下各节 |


### 4.2 响应体 — 所有题型共用的顶层字段（必须出现）


| 字段           | 类型      | 必须                   | 说明                                                              |
| ------------ | ------- | -------------------- | --------------------------------------------------------------- |
| `request_id` | string  | **是**                | **必须与请求完全一致**                                                   |
| `session_id` | string  | 否                    | `**app.py` 成功/失败响应均不返回该字段**；平台判分未校验响应内 `session_id`；自研网关可回显也可省略 |
| `task_type`  | string  | **是**                | **必须与请求 `task_type` 一致**（平台大小写不敏感比对）；不一致 → **整题失败**             |
| `ok`         | boolean | **是**                | `true` 表示进入 `result` 结构化校验与判分；`false` 表示本抽样失败（**仍须返回合法 JSON**）  |
| `result`     | object  | null                 | **是**                                                           |
| `message`    | string  | **是**                | 失败原因；成功时 **必须为** `""`（空字符串）                                     |
| `elapsed_ms` | number  | **是**（与 `app.py` 一致） | 整数毫秒；`app.py` 在成功与失败分支均返回                                       |


**说明**：平台从响应中取 `**result` 对象**送判分器；若顶层缺少 `result` 键，会把**根对象**当作 `result`（兼容旧实现）。`**app.py` 始终包含 `result` 键。**

### 4.3 与 `app.py` 一致的 `meta.extra`（联调可选）

`app.py` 会读取 `**meta.extra`**（若存在且为对象）：


| 字段            | 类型      | 说明                                           |
| ------------- | ------- | -------------------------------------------- |
| `delay_ms`    | number  | 本次推理前睡眠毫秒数；缺省为环境变量 `MOCK_DELAY_MS` 或 **120** |
| `force_error` | boolean | 为 `true` 时强制返回 `ok: false`（见下失败响应示例文案）       |


**说明**：若 `meta.extra` 存在但**不是**对象，`app.py` 视为未配置：延迟用 `MOCK_DELAY_MS` 或 **120**，`force_error` 为 `false`。

---

## 五、`task_type: "ocr"` — 请求与返回（固定示例）

### 5.1 请求体示例（完整 JSON）

**与 `app.py` 中 `mock_ocr_result` 一致的行为（占位文本来源）**：

- 先取 `meta.expected.text`（当 `meta.expected` 为对象且 `text` 为字符串时写入中间变量）；若存在 `meta.samples[0].expected.text`（`samples` 为非空列表且首元素为含 `expected` 的对象），则用其**覆盖**该中间变量。
- 上述合并结果 `strip()` 后非空，则返回 `{"text": "<该字符串>"}`。
- 否则对 `image.data` 做 SHA-256，取摘要前 **8** 个十六进制字符大写为 token，返回 `{"text": "DEMO-<TOKEN>"}`。

```json
{
  "request_id": "eval-1205-1-1",
  "session_id": "team-7",
  "task_type": "ocr",
  "image": {
    "format": "url",
    "data": "https://platform.example.com/files/signed/abc123.jpg"
  },
  "meta": {
    "difficulty": "L2",
    "sample_index": 1,
    "sample_k": 1,
    "infer_T_max_ms": 15000,
    "language_hint": "zh",
    "normalize_rules": {
      "trim_space": true,
      "case_insensitive": false
    },
    "draw_source": "admin-bank"
  }
}
```

`**result` 内字段（固定）**


| 字段        | 必须              | 说明                                                             |
| --------- | --------------- | -------------------------------------------------------------- |
| `text`    | **约定主字段（必须实现）** | 识别结果字符串；平台与标准答案做同一套 `normalize_rules` 后 **全等**比对               |
| `content` | 否               | 仅当未提供 `text` 时平台会读取 `content`；**交付时必须有 `text`，不得依赖 `content`** |


### 5.2 响应体示例（与 `app.py` 成功分支字段顺序一致：`request_id` → `task_type` → `ok` → `result` → `elapsed_ms` → `message`）

```json
{
  "request_id": "eval-1205-1-1",
  "task_type": "ocr",
  "ok": true,
  "result": {
    "text": "阀位12.8%"
  },
  "elapsed_ms": 85,
  "message": ""
}
```

### 5.3 `ok: false` 时响应示例（与 `app.py` 中 `meta.extra.force_error` 分支一致）

```json
{
  "request_id": "eval-1205-1-1",
  "task_type": "ocr",
  "ok": false,
  "result": null,
  "elapsed_ms": 128,
  "message": "按 meta.extra.force_error 指令返回模拟失败"
}
```

（`elapsed_ms` 为 `time.perf_counter()` 换算的整数毫秒，**含** `meta.extra.delay_ms`（或默认 `MOCK_DELAY_MS` / 120）睡眠耗时，故通常 ≥ 配置的延迟；上式中 `128` 仅为示意。）

---

## 六、`task_type: "classify"` — 请求与返回（固定示例）

### 6.1 请求体示例（完整 JSON）

```json
{
  "request_id": "eval-1205-1-1",
  "session_id": "team-7",
  "task_type": "classify",
  "image": {
    "format": "url",
    "data": "https://platform.example.com/files/signed/def456.jpg"
  },
  "meta": {
    "difficulty": "L1",
    "sample_index": 1,
    "sample_k": 1,
    "infer_T_max_ms": 15000,
    "class_names": ["正常", "告警", "离线"],
    "draw_source": "admin-bank"
  }
}
```

`**result` 内字段（固定主写法）**


| 字段      | 必须                 | 说明                                                                                 |
| ------- | ------------------ | ---------------------------------------------------------------------------------- |
| `label` | **是**（**固定使用本字段**） | 单标签字符串；**必须**落在当次 `meta.class_names` 内（大小写不敏感），否则该抽样 **0 分**（`invalid_label_name`） |


**与 `app.py` 中 `mock_classify_result` 一致**：若 `meta.class_names` 为列表，先将其转为字符串列表作为候选；若存在 `meta.samples[0].meta.class_names` 且为列表，则**用其替换**整个候选列表（与代码顺序一致）。候选非空时，用 `stable_random(image.data)`（SHA-256 摘要前 16 个十六进制字符转整数为 `random.Random` 种子）从中选一项返回。候选仍为空时，从固定池 `**["normal", "alarm", "offline"]`** 中同样方式选一项。§6.1 请求示例中的中文类名仅演示平台下发形状，与无类名时的英文池无关。

**平台仍兼容读取的别名字段（非本约定主写法）**：`label_id`、`prediction`、`class`；**新开发只输出 `label` 即可。**

### 6.2 响应体示例（与 `app.py` 成功分支一致）

```json
{
  "request_id": "eval-1205-1-1",
  "task_type": "classify",
  "ok": true,
  "result": {
    "label": "正常"
  },
  "elapsed_ms": 42,
  "message": ""
}
```

---

## 七、`task_type: "detect"` — 请求与返回（固定示例）

### 7.1 坐标系（固定）

- `**meta.coord_mode` 必须为字符串 `"pixel"**`（在平台下发的 `meta` 中；根或样本合并后一致）。  
- **评委框 `expected.boxes[].xyxy` 与选手返回的坐标全部为像素**，原点为**题图左上角**，x 向右、y 向下，与**该题图自然宽高**一致。  
- **禁止使用 0～1 归一化坐标**；`coord_mode` 非 `pixel` 时平台判分 **0 分**或入库校验失败。

### 7.2 请求体示例（完整 JSON）

```json
{
  "request_id": "eval-1205-1-1",
  "session_id": "team-7",
  "task_type": "detect",
  "image": {
    "format": "url",
    "data": "https://platform.example.com/files/signed/ghi789.jpg"
  },
  "meta": {
    "difficulty": "L2",
    "sample_index": 1,
    "sample_k": 1,
    "infer_T_max_ms": 15000,
    "coord_mode": "pixel",
    "scoring": "center_in_box",
    "class_names": ["defect", "valve"],
    "image_width": 640,
    "image_height": 480,
    "draw_source": "admin-bank"
  }
}
```

说明：`image_width` / `image_height` 为题面参考尺寸（若平台下发）；**判分几何以像素 xyxy / cx,cy 与题库 `expected` 一致为准**。

### 7.3 `result` 内结构（与 `app.py` 一致：`mock_detect_targets_result`）

**固定**：`result` 为对象，且含 `**targets`** 非空数组。每项**必须**含：


| 字段      | 必须    | 说明                                                         |
| ------- | ----- | ---------------------------------------------------------- |
| `label` | **是** | 字符串；`app.py` 在存在 `meta.class_names` 时从中随机选一，否则为 `"defect"` |
| `cx`    | **是** | 浮点数；`app.py` 为 `80～560` 内随机整数转 float（像素 x）                 |
| `cy`    | **是** | 浮点数；`app.py` 为 `60～420` 内随机整数转 float（像素 y）                 |
| `score` | 否     | `app.py` 会附带 `0.9～0.99` 之间三位小数；平台判分不强制校验                   |


**说明**：竞赛平台后端另接受 `**boxes` / `detections`** 等结构（见 `VisualScoringService`）；**本仓库参考实现 `app.py` 仅实现上述 `targets` 形状**，文档以此为准。

### 7.4 响应体示例（与 `app.py` 成功分支、`mock_detect_targets_result` 一致）

```json
{
  "request_id": "eval-1205-1-1",
  "task_type": "detect",
  "ok": true,
  "result": {
    "targets": [
      {
        "label": "defect",
        "cx": 320.0,
        "cy": 240.0,
        "score": 0.942
      }
    ]
  },
  "elapsed_ms": 118,
  "message": ""
}
```

（`cx`/`cy`/`label`/`score` 的具体数值由 `app.py` 依 `image.data` 与 `meta` 随机生成，上表仅为形状示例。）

### 7.5 不支持 `task_type` 时（与 `app.py` 分支一致）

```json
{
  "request_id": "eval-1205-1-1",
  "task_type": "keypoint",
  "ok": false,
  "result": null,
  "elapsed_ms": 125,
  "message": "不支持的 task_type: keypoint，当前示例仅支持 ['classify', 'ocr', 'detect']"
}
```

（`message` 与 `app.py` 在未知 `task_type` 分支（约第 87–95 行）生成的字符串格式一致；`elapsed_ms` 同样**含**前置 `delay_ms` 睡眠，非固定值，上式为示意。）

---

## 九、交付自检清单（必须满足）

- `GET /health`：**200** + JSON，`status === "ok"`，`supported_tasks` **包含**本队参赛所需的全部 `task_type`  
- `POST /infer`：三种题型均能按上文章节与 `**app.py`** 返回**完整信封**且 `result` 符合表结构  
- `request_id` / `task_type`：**响应与请求一致**（`session_id` 响应可省略，与 `app.py` 一致）  
- `**classify`**：`result.label` **属于**当次 `meta.class_names`  
- `**detect`**：与 `app.py` 一致时仅返回 `**result.targets`**（像素 `cx`/`cy`）；平台另支持 `boxes` 等见后端实现  
- `**ocr**`：`result.text` **存在**且为字符串  
- 跨域、鉴权头、超时已在真实浏览器环境下验证通过

---

**文档版本**：**JSON 形状与字段以 `初赛竞赛系统/选手算法服务-test/app.py` 为准**；平台判分扩展行为见 `VisualScoringService`、`ClientEvaluationResultApplier`。