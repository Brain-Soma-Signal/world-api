# World API

Soma & Signal 的环境交互契约。模型收到一帧公开观察，提交动作，再读取环境反馈。使用同一接口可以重放一条轨迹，也可以接入另一套环境实现。

[阅读观察与动作协议](v0.1/CONTRACT.md) · [查看一条完整轨迹](v0.1/examples/transcript.jsonl)

```sh
cd v0.1
python -m virtual_world_api.demo
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

v0.1 保留 Virtual_World 接口包的原始文件、示例环境与 SHA-256 清单。协议变更采用新的版本目录。

[Soma & Signal 组织首页](https://github.com/Brain-Soma-Signal)
