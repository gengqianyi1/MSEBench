# Task 01 — 安全隔离证据

## Docker 挂载隔离

| 挂载路径 | 权限 | 说明 |
|---------|------|------|
| `workspace/` | 可读写 | Agent 唯一工作目录 |
| `data/*.json.gz` | 只读 | 公开数据集 |
| `tasks/` | 只读 | 公开题目配置 |
| `private/` | — | **不挂载**（.dockerignore 排除） |
| `runs/` | — | **不挂载**（.dockerignore 排除） |

## 路径越界防护

`src/tool_executor.py` `_validate_path()` 拦截：
- `../` 上溯
- 绝对路径（如 `/etc/passwd`, `C:\`）
- `~` 展开
- 符号链接指向 workspace 外的路径

## 网络隔离

| 层级 | 配置 |
|------|------|
| Docker | `--network none`（评价运行时） |
| 工具执行 | `configs/default.yaml` `allow_network: false` |
| LLM API | 由 orchestrator 在外层发起，不经过 Agent 工作容器 |

## 拒绝访问测试

`tests/test_tool_executor.py` 包含：
- `test_reject_parent_traversal`
- `test_reject_absolute_path`
- `test_reject_private_dir_read`
