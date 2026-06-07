# 中控杯初赛本地算法服务

这个仓库提供“工业多模态感知与无人化智能操作”初赛的本地算法服务工程：`GET /health`、`POST /infer`、三类任务 fallback backend、日志、测试、打包脚本和备赛文档都已经就位。当前默认是离线优先、可联调、可替换真实模型的版本。

## 规则基线

- 实现以 `docs/rules/选手端算法说明文件.pdf` 和 `docs/rules/赛题2（外操无人化）初赛通知.pdf` 为准。
- 当前赛题标签：
  - `classify`: `办公室、公园、街道、商场、厨房、卧室、图书馆、体育馆`
  - `detect`: `人、汽车、自行车、手机、水杯、笔记本电脑、台灯、沙发、狗`
- `detect` 坐标固定为原图 `pixel` 坐标，判分按“类别一致 + 中心点落入参考框”。
- 禁止外部云服务、大模型 API、商业接口代替本地推理。

## 环境要求

- Python `3.10+`
- 建议 Windows PowerShell、Git Bash 或 Linux/macOS Shell

## 安装

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含 `-e .`，安装后可以直接从仓库根目录运行 `contest_agent` 包。

## 启动

推荐：

```bash
uvicorn contest_agent.app:app --host 0.0.0.0 --port 8080
```

或：

```bash
uvicorn --app-dir src contest_agent.app:app --host 0.0.0.0 --port 8080
```

或：

```bash
python -m contest_agent.app
```

Shell 脚本版本：

```bash
bash scripts/run_server.sh
```

## `/health` 测试

```bash
curl http://127.0.0.1:8080/health
```

期望返回：

```json
{
  "status": "ok",
  "supported_tasks": ["classify", "ocr", "detect"],
  "service": "contestant-algo-test-service",
  "version": "0.2.0",
  "bridge_mode": "local"
}
```

## `/infer` 三类任务测试

快速联调：

```bash
bash scripts/smoke_test.sh
```

Windows 直接运行：

```bash
python scripts/smoke_test.py
```

也可以直接发本地图片：

```bash
curl -X POST http://127.0.0.1:8080/infer \
  -H "Content-Type: application/json" \
  -d "{\"request_id\":\"demo-1\",\"session_id\":\"team-7\",\"task_type\":\"ocr\",\"image\":{\"format\":\"path\",\"data\":\"tests/fixtures/sample.jpg\"},\"meta\":{\"normalize_rules\":{\"trim_space\":true,\"case_insensitive\":false}}}"
```

## 配置项

见 `.env.example`，核心项如下：

- `APP_HOST` / `APP_PORT`: 服务监听地址
- `OFFLINE_MODE=true`: 默认离线运行
- `MODEL_CLASSIFY_PATH` / `MODEL_DETECT_PATH` / `MODEL_OCR_PATH`: 本地模型路径
- `CLASSIFY_BACKEND` / `DETECT_BACKEND` / `OCR_BACKEND`: `fallback` 或 `local`
- `DETECT_SCORE_THRESHOLD`: 检测置信度阈值预留项
- `DETECT_EMPTY_FALLBACK`: 检测无结果时是否补一个保底点
- `IMAGE_DOWNLOAD_TIMEOUT_SECONDS`: 题图下载超时
- `MAX_IMAGE_SIZE_MB`: 图像大小限制
- `LOG_FILE`: 日志输出路径

## 模型权重放置说明

- 把真实权重放到 `models/`。
- 当前代码已预留三个本地后端入口：
  - `src/contest_agent/inference/classifier.py`
  - `src/contest_agent/inference/detector.py`
  - `src/contest_agent/inference/ocr.py`
- 若权重不存在，服务会自动降级为 fallback，不会阻塞启动。

## 离线模式说明

- 默认 `OFFLINE_MODE=true`。
- 离线模式并不阻止读取比赛平台下发的题图 URL，因为这是比赛流程的一部分。
- 代码没有接入任何外部云推理 API。

## 测试

```bash
pytest -q
```

覆盖点包括：

- `/health` 基础契约
- `/infer` 三类任务成功路径
- 错误 `task_type`
- 缺字段时 JSON 错误返回
- `OPTIONS` 预检和 CORS
- `detect` 像素坐标与英文标签映射
- 图片读取失败路径
- `request_id` 原样回显

## 打包

```bash
bash scripts/package_submission.sh
```

Windows 直接运行：

```bash
python scripts/package_submission.py
```

输出：

- `dist/contest_agent_submission.zip`
- `dist/contest_agent_submission.zip.sha256`

脚本会清理缓存、排除 `.venv`、统计模型大小，并检查压缩包是否超过 `2GB`。

## 比赛当天运行流程

1. `pip install -r requirements.txt`
2. 放置本地权重到 `models/`
3. 启动服务：`uvicorn contest_agent.app:app --host 0.0.0.0 --port 8080`
4. 自测：`curl http://127.0.0.1:8080/health`
5. 烟测：`bash scripts/smoke_test.sh`
6. 平台 Agent Base URL 只填根地址，例如 `http://127.0.0.1:8080`

## 常见故障排查

- 端口占用：改 `APP_PORT` 或结束占用进程。
- CORS 失败：确认浏览器预检 `OPTIONS` 能返回，且 `/health`、`/infer` 都带跨域头。
- 模型权重缺失：查看 `logs/app.log`，服务会降级 fallback。
- 图片 URL 读取失败：检查网络、签名 URL 是否过期、是否超出 `IMAGE_DOWNLOAD_TIMEOUT_SECONDS`。
- 坐标归一化错误：`detect` 必须输出原图像素 `cx/cy`，不能是 `0~1`。
- `label` 不在 `class_names`：检查中文标签映射，最终输出必须落在合法集合内。
- 超时：关注 `meta.infer_T_max_ms`，必要时切小模型或增加预处理缓存。
- 非 JSON 响应：所有异常都必须走统一 JSON 错误信封。

## 目录结构

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── scripts/
├── src/contest_agent/
├── tests/
├── docs/
├── models/
└── logs/
```

## 复现说明

- 保留 `tests/`、`scripts/`、`docs/` 与必要模型权重一起打包。
- 赛后 3 小时内提交，压缩包不得超过 `2GB`。
- 如果后续接入真实模型，请同步记录权重版本、预处理、后处理和运行命令，保证可复现。
