# Agent Notes

1. 默认离线运行，`OFFLINE_MODE=true`，不要引入任何外部云推理 API。
2. 允许访问比赛平台下发的题图 URL，但禁止把题图发到外部推理服务。
3. 每次修改后优先运行 `pytest -q`，不要破坏 `/health` 和 `/infer` 的 JSON 契约。
4. `GET /health` 必须返回 `status: "ok"`，`supported_tasks` 至少覆盖 `classify`、`ocr`、`detect`。
5. `POST /infer` 失败时也要返回合法 JSON，不能返回 HTML、纯文本或抛未捕获异常。
6. `detect` 坐标必须是原图 `pixel` 坐标，判分关注中心点命中，不是 IoU。
7. `classify` 和 `detect` 输出必须是合法中文标签，优先以 `meta.class_names` 为准。
8. `ocr` 必须输出 `result.text`，后处理要保守，避免误伤有效字符。
9. 模型加载要懒加载或弱依赖，权重缺失时必须降级 fallback，服务仍可启动。
10. 新增依赖必须同步写入 `requirements.txt`；大模型权重不要提交到 Git，比赛打包时再放到 `models/`。
