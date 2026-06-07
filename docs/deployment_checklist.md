# 部署检查清单

## 环境与依赖

- [ ] Python 版本符合要求，建议 `python --version`
- [ ] 已执行 `pip install -r requirements.txt`
- [ ] `models/` 里的权重路径与 `.env` 配置一致
- [ ] `OFFLINE_MODE=true`

## 启动服务

- [ ] 启动：`uvicorn contest_agent.app:app --host 0.0.0.0 --port 8080`
- [ ] 或备用：`uvicorn --app-dir src contest_agent.app:app --host 0.0.0.0 --port 8080`

## 健康检查

- [ ] `curl http://127.0.0.1:8080/health`
- [ ] 返回 `status=ok`
- [ ] `supported_tasks` 包含 `classify`、`ocr`、`detect`

## 三类 `/infer` 自测

- [ ] `bash scripts/smoke_test.sh`
- [ ] classify 返回合法中文 `label`
- [ ] detect 返回 `targets[].label/cx/cy`
- [ ] ocr 返回 `result.text`

## 浏览器与 CORS

- [ ] 浏览器可访问平台页面
- [ ] `OPTIONS /infer` 预检可通过
- [ ] `/health`、`/infer` 都带 CORS 允许头

## 网络与端口

- [ ] 队长机防火墙未阻止服务端口
- [ ] 平台 Agent Base URL 只填根地址，例如 `http://127.0.0.1:8080`
- [ ] 不要在平台里填写 `/infer`

## 日志与异常

- [ ] 查看 `logs/app.log`
- [ ] 关注 `request_id`、`task_type`、`elapsed_ms`、`image size`、`message`
- [ ] 确认失败路径返回的也是合法 JSON

## 打包

- [ ] 运行 `bash scripts/package_submission.sh`
- [ ] 检查 zip 文件生成成功
- [ ] 检查 zip 未超过 `2GB`
- [ ] 检查 `.sha256` 文件已生成

## 备用方案

- [ ] 备用端口已准备
- [ ] 备用启动命令已验证
- [ ] fallback backend 在权重缺失时仍能跑通接口
