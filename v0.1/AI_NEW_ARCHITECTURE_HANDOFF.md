# 世界接口 v0.1：AI-New_Architecture 对接说明

接口版本：`0.1.0`。本包提供纯 Python 包 `virtual_world_api`、数据约定及原创 mock；mock 是验证调用顺序的小夹具，不是从小说构建完成的世界。首版为单主体、有限离散动作，不依赖 Torch，不加载模型，也不启动架构项目的实验。

## 1. 两个项目各自负责什么

世界端负责状态演化、公开观察、合法动作、已发生反馈和终止条件。模型端负责文本编码、动作选择、FusedQwen 状态、反馈调制规则及训练。runner 连接两者并记录日志。

模型只能接收 `observation`、`actions` 和已经发生的 `last_feedback`，以及由这些公开信息组成的历史。不要把 `env`、完整请求、随机种子、隐藏规则、未来事件或评估真值传入模型。`world_id`、`episode_id` 等是运行管理元数据，默认不进入提示词。

`context_id` 由模型端选择，例如固定主体的记忆槽或某次独立评估的状态命名空间；它不自动等于 `world_id`。环境 `reset()` 不会重置模型。跨世界保留、清零或保存恢复哪些状态，须由实验协议明确指定。

## 2. 冻结的数据与调用约定

请求和返回值均为 dataclass，使用 `.to_dict()` 得到可序列化字典；不要依赖内部 `__dict__`。接口如下：

```python
WorldEnv.reset(ResetRequest) -> Frame
WorldEnv.observe() -> Frame
WorldEnv.step(ActionRequest) -> Frame
WorldEnv.close()

ResetRequest(
    world_id, world_version, episode_id, seed, max_steps,
    reveal_world_id=True,
)
ActionRequest(episode_id, expected_step, request_id, action_id)
```

`ResetRequest` 只供 runner 使用。关闭世界身份公开时，Frame 的 `world_id` 与 `world_version` 为 `None`，但模型桥的状态管理仍由 consumer 显式完成。

```text
Frame:
  schema_version: "0.1.0"
  world_id: str | None
  world_version: str | None
  episode_id: str
  step_index: int
  observation: {text: str, facts: {公开字段名: JSON值, ...}}
  actions: [{id: str, label: str}, ...]
  last_feedback: None | {
    transition_id: str,
    from_step: int,
    action_id: str,
    reward: float | None,
    events: [str, ...]
  }
  terminated: bool
  truncated: bool
```

`facts` 是 JSON 对象，具体字段按世界的公开观察定义，桥接器应完整保留，不另造隐藏字段。动作只能引用当前 `actions` 中的 `id`，不接受自由文本命令或额外动作参数。`expected_step` 必须对应作出该动作的观察。`last_feedback` 描述刚完成的转换；首次观察为 `None`。`terminated` 表示世界终止，`truncated` 表示步数等外部限制，二者都停止本回合的动作循环。

重复 `observe()` 不推进世界，也不代表新经历，不能因此再次提交模型状态。消费反馈时同时核对 `episode_id`、`from_step`、`action_id` 和 `transition_id`，防止把旧奖励写入新动作的资格迹。

## 3. FusedQwen 的最小接线

截至 2026-10-29，阶段45的旧最后层插件迁移已失败并封存：合并准确率48.4375%，base为50%。阶段47完成196投影迁移工程，19次前向、1次反向、0次优化更新；工程连通不等于已经获得小说世界能力。当前应参考 [FusedQwen 控制器](../../../AI-New_Architecture/src/selflab/fused_qwen.py)、[FusedLinear 单元](../../../AI-New_Architecture/src/selflab/fused_unit.py) 和 [最新研究状态](../../../AI-New_Architecture/research/RESEARCH_STATUS.md)。旧最后层插件说明仍可作历史材料，不能当作最新结构。

事务顺序是：`begin → forward → commit → apply_feedback / waive_feedback`。以下为未执行的接线伪代码；`model/controller` 由模型项目建立，`encode_public`、`choose_legal_action`、`validate_feedback_matches`、`map_public_delta` 及 `journal` 是 consumer 需实现的接口。此例只做推理，不调用优化器。

```python
import torch
from uuid import uuid4
from virtual_world_api import ActionRequest

# runner 已调用 frame = env.reset(reset_request)
# context_id 由实验协议选择，不从 world_id 自动推导。
while not (frame.terminated or frame.truncated):
    data = frame.to_dict()
    public = {k: data[k] for k in ("observation", "actions", "last_feedback")}
    tokens = encode_public(public)  # 只编码公开信息
    mask = tokens["attention_mask"].to(device="cpu", dtype=torch.bool)
    event_id = uuid4().hex
    before_state = controller.export_state()  # 空闲状态，供明确失败恢复

    controller.begin([context_id], event_id, mask, training=False)
    try:
        with torch.inference_mode():
            output = model(**tokens, use_cache=False)
        action_id = choose_legal_action(output, public["actions"])
        if action_id not in {a["id"] for a in public["actions"]}:
            raise ValueError("Action is not available in this frame")
        receipts = controller.commit()
    except Exception:
        controller.rollback()  # 本块失败时事务仍处于 active 状态
        raise

    request = ActionRequest(
        episode_id=frame.episode_id,
        expected_step=frame.step_index,
        request_id=uuid4().hex,
        action_id=action_id,
    )
    journal.before_dispatch(
        request.to_dict(), event_id, receipts,
        before_state, controller.export_state(),
    )  # 持久化：世界调用异常后，不得重新抽动作
    next_frame = env.step(request)
    feedback = next_frame.to_dict()["last_feedback"]
    journal.record_transition(next_frame.to_dict())
    validate_feedback_matches(request, feedback)

    delta = map_public_delta(feedback)  # 无授权公开调制信号时返回 None
    if delta is None:
        controller.waive_feedback(receipts)
    else:
        controller.apply_feedback(receipts, {context_id: delta})
    journal.after_feedback(controller.export_state())
    frame = next_frame
```

consumer 的 `choose_legal_action` 必须输出动作ID，而非把整段生成文字直接交给环境。第一版推荐一次“观察＋合法动作”前向，并在约定的动作符号中选择。若协议允许整段文本生成，需要另行确定token边界、合法动作解析和唯一提交位置；这里没有宣称 `model.generate()` 可直接接入。

## 4. 反馈、状态与故障恢复

- **reward 与 delta 分开。** `reward` 是世界提供的公开结果；`delta` 是模型协议定义的调制量，必须为有限数、非bool、处于 `[-1,1]`。不要未经说明直接裁剪任意奖励，也不要利用隐藏正确答案构造delta。没有公开奖励仍可能有公开事件，但是否可据此产生delta须另定规则。
- **缺反馈不等于0。** `apply_feedback(..., delta=0)` 仍执行 `w ← 0.99w`；`waive_feedback()` 消费receipt而不更新w。`reward=None` 不能默认为0。反馈只影响下一事务；不能先取得本动作奖励再决定本动作。
- **旧二元更新不能混用。** 旧插件的 `y=(2a−1)(2r−1)`、±1刺激和8端口是特定任务公式。本世界接口不承诺这些条件，新FusedLinear的delta接口也不是已经验证有效的通用RL算法。
- **四上下文与批处理。** 当前每个FusedLinear最多保存4个context；同一batch中ID必须不同。上一receipt必须消费或明确放弃后，才可在该context开始新事务。单主体首版可使用batch=1。清空会清除整个控制器的状态库，不是单独淘汰一个世界；超过容量前应按协议保存/清空/恢复。
- **一次提交对应一次经历。** 事务内forward读取同一已提交状态，但会把pending覆盖为最近一次forward的最后有效token。逐候选评分后直接commit会写入最后枚举的候选，而非所选行动。多分支评估须回滚分支，并为唯一规范输入重新前向后提交，不能给每个候选各提交一次。
- **缓存与图。** v0桥使用 `use_cache=False`。训练实现需遵守最多两个训练事务的图边界，并在优化器更新后调用 `detach_state()`；快状态保存须在事务外，恢复须为空闲且已detach。环境JSON与模型张量检查点分别保存，通过事件ID关联。

恢复分三种情况处理：

1. **模型forward或提交失败，尚未调用世界：** 回滚活动事务，世界不变。修正错误后重新开始，不消费反馈。
2. **`step`异常或返回丢失，是否执行不明：** 保留原 `ActionRequest`，以完全相同的 `request_id` 和全部字段重试。不要重新选择动作、改 `expected_step` 或再提交模型状态。世界端必须将同一请求返回为同一转换；同ID不同内容应拒绝且不写入。幂等缓存检查须先于“当前step已变化”检查。该恢复保证限于环境仍保留请求账本的同一回合；不能假定重启或reset后账本自动存在。
3. **明确拒绝且确认世界未推进，或消费反馈时失败：** 前者恢复日志中的 `before_state`，再处理拒绝原因；后者保留已完成世界转换，只恢复/完成对应receipt消费。若反馈实际已消费，不得再次应用。进程重启须恢复配对的世界状态/请求账本与模型状态；无法确认时中止该回合并记录，不能用 `reset()` 冒充续接。

每次成功转换都应记录请求、返回Frame、公开反馈到delta的映射版本、模型事务ID以及检查点引用。先用本包mock验证通信、幂等与边界，再由另一项目依其研究协议决定是否开展功能实验；本交接本身没有启动模型训练。
