# MSEBench 本地评测入口

安装与镜像部署见 [DEPLOY.md](DEPLOY.md)，用户操作见 [运行指南](docs/RUN_GUIDE.md)，导出流程见 [公开发布流程](PUBLIC_TASK_DEFINITIONS.md)。

在组装完成的工作目录执行。

```bash
python -m pip install '.[local]'
python scripts/validate_tasks.py
python run.py --task textqa --verify
msebench --workspace "$PWD" --task 4 --verify
msebench --workspace "$PWD" --task 4
```

模型地址、名称和密钥通过 MSEBENCH_API_BASE、MSEBENCH_MODEL_ID、MSEBENCH_API_KEY 注入。材料学任务的评分器在用户本地 verifier 容器中运行。
