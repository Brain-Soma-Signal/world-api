# 世界模型环境接口 v0.1.0

交付日期：2026-10-29。供 `Virtual_World` 和 `AI-New_Architecture` 对接的首个稳定环境协议。实现仅使用 Python 3.9+ 标准库，不依赖模型框架或网络服务。

本包交付接口、可重放 mock、JSON Schema、示例和契约测试。小说目前是素材库，**尚未实现小说世界引擎**。`demo.ledger`、`demo.workshop`、`demo.dream` 是同一原创二选一机制的三套表面描述，不能据此声称完成三个独立世界或意识实验。

- [协议与字段](CONTRACT.md)
- [给架构项目的接线说明](AI_NEW_ARCHITECTURE_HANDOFF.md)
- [JSON Schema](world.schema.json)
- [Python接口](virtual_world_api/api.py)
- [最小运行示例](virtual_world_api/demo.py)
- [JSON请求、观察及轨迹](examples/)
- [测试结果](VALIDATION.json)
- [冻结文件清单与SHA-256](MANIFEST.json)

## 立即运行

在本包目录执行（不加载任何模型）：

```bash
python3 -m virtual_world_api.demo
python3 -m unittest discover -s tests -v
```

在本仓库根目录执行：

```bash
PYTHONPATH=interfaces/world-v0.1 python3 -m virtual_world_api.demo
```

Schema验证需要可选开发依赖 `requirements-dev.txt`。未安装时，核心契约测试仍可运行，Schema测试明确显示 skipped。

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m unittest discover -s tests -v
```

## 最小调用

```python
from uuid import uuid4
from virtual_world_api import ActionRequest, ResetRequest
from virtual_world_api.mock import DemoWorld

env = DemoWorld()
try:
    frame = env.reset(ResetRequest(
        world_id="demo.ledger", world_version="1.0.0",
        episode_id=str(uuid4()), seed=7, max_steps=8,
    ))
    # 将 frame 的公开内容交给模型；不将 env、seed 或私有评测信息交给它。
    request = ActionRequest(frame.episode_id, frame.step_index,
                            str(uuid4()), "inspect")
    next_frame = env.step(request)
    assert env.step(request) == next_frame  # 相同请求重试不会重复执行
    print(next_frame.to_dict())
finally:
    env.close()
```

`Frame(t)`含当前观察及上一动作的反馈；`step(a_t)`返回`Frame(t+1)`。模型状态由接收项目管理。`reward=None`表示没有公开数值奖励，`reward=0.0`表示实际观测到零奖励；世界奖励与模型调制量delta的映射需单独定义。

`DemoWorld.private_trace()`仅供runner/评测器记录种子、隐藏目标和事件重放，未列入公开`WorldEnv`协议。模型不应获得环境对象。这是程序调用边界，不是对不可信Python代码的安全沙箱。

接口版本`0.1.0`与mock世界版本`1.0.0`分别标识。该目录在交付manifest后冻结；后续变更另建版本，不原地修改接收项目已经固定的版本。可将同名zip解压到任意目录并将包根加入`PYTHONPATH`，不需要复制小说语料。另一项目的FusedQwen链接针对当前同级工作区布局；异地使用时请按其项目位置调整。
