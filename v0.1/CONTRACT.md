# 虚拟世界环境对接合同 v0.1.0

本合同规定单主体、离散动作的同步 Python 环境边界。环境提供规则世界及公开反馈；消费方项目负责模型推理、记忆管理和实验调度。配套 mock 是用于联调的规则世界，不代表已从小说构建真实世界。

运行接口使用 Python 3.9+ 标准库，不需要 HTTP 服务或第三方训练框架。本版不实现 RPC、checkpoint、世界恢复或延迟反馈。

## 接口与数据

```text
WorldEnv.reset(request: ResetRequest) -> Frame
WorldEnv.observe() -> Frame
WorldEnv.step(request: ActionRequest) -> Frame
WorldEnv.close() -> None
```

请求和返回值使用 dataclass；`to_dict()` 返回独立的、可序列化为 JSON 的数据。JSON 结构见 [world.schema.json](world.schema.json)，采用 Draft 2020-12：文件顶层校验 `Frame`，请求定义为 `$defs/ResetRequest` 与 `$defs/ActionRequest`。所有合同对象拒绝额外字段；只有 `observation.facts` 是允许扩展的 JSON 对象。

标识符必须是非空字符串，最长 128 个字符，包括世界、版本、episode、request、action 和 transition 标识符。标识符是 opaque 值，消费方不能通过拆解它们推断规则或含义。所有 JSON 数字必须有限，不允许 NaN 或 Infinity；Python 的 `bool` 不能代替整数或奖励数值。`facts` 的键必须是字符串，值只能是 JSON 值，不得放入 Python 对象、集合或循环引用。

| 数据 | 字段 |
| --- | --- |
| `ResetRequest` | `world_id: str`、`world_version: str`、`episode_id: str`、`seed: int`、`max_steps: int`、`reveal_world_id: bool = True` |
| `ActionRequest` | `episode_id: str`、`expected_step: int`、`request_id: str`、`action_id: str` |
| `Frame` | `schema_version: "0.1.0"`、`world_id: str 或 null`、`world_version: str 或 null`、`episode_id: str`、`step_index: int`、`observation`、`actions`、`last_feedback`、`terminated: bool`、`truncated: bool` |
| `observation` | `text: str`、`facts: JSON object` |
| `actions` 中的单个动作 | `id: str`、`label: str` |
| 非空 `last_feedback` | `transition_id: str`、`from_step: int`、`action_id: str`、`reward: number 或 null`、`events: list[str]` |

`seed` 范围为 0 至 4294967295；`max_steps` 为 1 至 100000；步号为 0 至 100000。`ResetRequest` 的 JSON 可省略 `reveal_world_id`，消费方应在构建请求时采用默认值 `True`；Schema 中的 `default` 本身不会替调用方填入字段。其余请求字段和全部 `Frame` 字段均须存在。

## 生命周期

1. 创建环境后，先调用 `reset`，再调用 `observe` 或 `step`。
2. `reset` 要求已注册的 `world_id`、完全匹配的 `world_version`，以及该环境对象生命周期内未成功使用过的新 `episode_id`。成功后初始化新世界动态状态和环境 RNG，步号为 0，`last_feedback=null`，两个结束标志均为 `False`。旧 episode 的动作去重缓存清空，但已用 episode ID 的记录保留。被拒绝的 reset 不占用 episode ID，也不改变现有 episode。
3. `observe` 返回当前 Frame 的深拷贝，不推进时间、不执行动作、不产生新反馈、不消费随机数。重复读取可见同一条 `last_feedback`，不是重复发生一次事件。
4. `step` 成功提交一个动作时，世界恰好推进一步，步号恰好加 1。所有已声明的请求错误均保持世界状态、RNG、步号、反馈和已接受请求缓存不变。
5. 任一结束标志为 `True` 后，仍可 `observe` 或重试已接受的同一请求；新的动作不再执行。继续运行需用新 episode ID 调用 `reset`。
6. `close` 释放环境资源并关闭对象；重复 `close` 无副作用。关闭后 `reset`、`observe`、`step` 返回 `CLOSED` 错误。

`reset` 总是开始新 episode，不隐含恢复此前访问的世界。A→B→C→A 中再次进入 A 仍是一个新 episode。是否保留代理的长期记忆、短期上下文、KV cache、递归状态或快速权重，由消费方实验条件决定；环境不得暗中修改这些模型状态。世界实例状态与代理状态不能混用。

## 动作、时序与幂等

令 `Frame(t)` 表示步号为 t 的返回值。它包含当前观察 `o_t`，以及上一动作 `a_(t-1)` 的结果；t=0 时没有上一反馈。提交 `expected_step=t` 的合法动作 `a_t` 后，返回 `Frame(t+1)`，其中：

```text
last_feedback.from_step == t
last_feedback.action_id == 已执行的 action_id
```

每次已提交的状态转移具有一个新的 `transition_id`，在本 episode 内唯一且重试时不变。消费方只能在首次收到该转移时更新一次自己的记忆或快速权重；可以用 `(episode_id, transition_id)` 去重。`events` 描述这次动作后的公开结果。`reward=null` 表示未提供奖励；`reward=0` 表示明确提供了零奖励。没有反馈延迟队列，不能把当前动作尚未产生的结果送入它的决策输入。

`actions` 中的 ID 在当前 Frame 内唯一。调用方从当前列表选择 ID，`label` 只用于展示；不得用下标代替 ID。允许提交不等于必定成功：例如一个合法的“尝试开门”可能得到失败反馈，这仍是已执行的一步。未知或当前未提供的动作属于 `INVALID_ACTION`，不消耗一步。动作列表不能利用尚未公开的隐藏条件提前透露成功与否。

`step` 对有效请求按以下顺序处理：

1. 检查当前 `episode_id`。旧 episode 请求返回 `EPISODE_MISMATCH`。
2. 在当前 episode 的已接受请求中查找 `request_id`。若所有请求字段都与首次提交相同，返回首次结果的深拷贝，不执行动作；若同 ID 对应不同 payload，返回 `REQUEST_CONFLICT`。
3. 未命中缓存时，检查 `expected_step` 是否等于当前步号，否则返回 `STALE_STEP`。
4. 检查 episode 是否已结束，如已结束则返回 `EPISODE_DONE`；随后检查动作合法性并执行。

对象关闭、尚未 reset 以及字段类型/范围错误，分别由 `CLOSED`、`NOT_READY`、`INVALID_REQUEST` 处理。重复请求查找必须早于“当前步号已过期”和“episode 已结束”的拒绝分支，否则正常的动作重试会失效。去重只覆盖已成功执行的动作；拒绝的请求不会占用 request ID。

## 公开信息与评测真值

`ResetRequest` 只由 runner 持有，不直接交给模型。尤其不得把 seed、私有世界状态、真实规则、正确动作、反事实奖励或数据划分拼进提示。

当 `reveal_world_id=True` 时，Frame 的 `world_id` 和 `world_version` 与请求一致；为 `False` 时，两者必须同时为 `null`。隐藏开关只影响公开元数据，不改变动力学、随机流或奖励；不能把被隐藏的 ID 再塞入 `facts` 或事件文本。

模型的输入适配器按白名单使用 `observation`、可选动作和允许公开的反馈；`episode_id`、步号和 request ID 是调度信息，不应未经设计直接进入模型。`facts` 不是私有状态的便捷出口，只能包含规则允许主体观察到的事实。评测真值由独立评测记录或内部评测接口管理，不在 Frame 中增加混合的 `info` 字段。

Schema 可以限制字段结构，不能证明事实内容没有泄漏。需要通过配对隐藏状态测试和输入日志检查验证这一边界。

## 结束与重放

`terminated=True` 表示规则世界自身的终点，如任务完成或世界内死亡。`truncated=True` 表示尚未自然结束、但本次运行的 `max_steps` 预算已耗尽。两者不能同时为 `True`；最后一个允许动作若同时达到自然终点，优先设置 `terminated=True, truncated=False`。终局 Frame 保留最终观察和最后动作反馈，且 `actions=[]`；不得自动 reset 或把截断冒充任务失败。

这一区分与强化学习环境常用的终止/截断语义一致；本合同另外明确了自然终止优先规则，消费方不能将两者无区别地用于后续学习目标。[Gymnasium 官方接口说明](https://gymnasium.farama.org/api/env/)

重放需要固定世界版本、引擎实现、seed、max_steps 和已接受动作序列。环境使用自己的随机流，与代理采样及模型噪声独立。读取、请求错误、重复请求和展示操作不能改变环境随机状态。同条件的重放应得到相同公开观察、动作集合、反馈与结束位置；新的 episode ID 和由它关联的转移标识符可以不同，不能把这些调度标签的差异误判为世界行为不同。

## 错误代码

错误以带有以下稳定 code 的 Python 异常报告，不返回伪造 Frame，也不将错误转换为动作奖励。本合同不规定网络错误封装。

| Code | 含义 |
| --- | --- |
| `INVALID_REQUEST` | 请求类型、字段、JSON 值或范围不合法 |
| `UNKNOWN_WORLD` | world_id 未注册 |
| `VERSION_MISMATCH` | 请求版本与已注册世界版本不符 |
| `EPISODE_REUSED` | 该环境对象已成功使用过 episode_id |
| `NOT_READY` | 尚未成功 reset |
| `EPISODE_MISMATCH` | 动作请求不属于当前 episode |
| `REQUEST_CONFLICT` | 已接受的 request_id 被用于不同 payload |
| `STALE_STEP` | 新请求的 expected_step 与当前步号不一致 |
| `INVALID_ACTION` | 新请求的 action_id 不在当前动作列表 |
| `EPISODE_DONE` | 请求在已结束的 episode 中执行新动作 |
| `CLOSED` | 环境对象已关闭 |

## 最小验收检查

- JSON值与Schema：先用 `virtual_world_api.api.validate_json_value` 拒绝非有限数和Python专属值，再校验Schema；`to_dict()`已包含JSON值检查。有效 reset/action/frame 可通过；额外字段、空或超长 ID、布尔值冒充整数被Schema拒绝。隐藏 ID 必须成对为 null；初始反馈为空，后续反馈非空；结束标志互斥且终局动作为空。
- 时序：初始步号 0；成功动作只增加一步；`from_step=step_index-1`；反馈归属与实际动作一致，transition ID 唯一。
- 纯读取：重复 observe 不消费 RNG；修改返回 Frame、嵌套 facts/actions 或 to_dict 结果，不改变环境及缓存。
- 幂等：相同请求重试只执行一次；终局动作也可重试；同 ID 不同 payload 拒绝；拒绝请求不占用 ID。
- 拒绝路径：错误 episode、错误步号、非法动作、重复 episode ID、未知世界和版本不符，均不改变既有 episode 及随机流。
- 终局：自然终止、预算截断、最后一步同时成功、结束后新动作和关闭后调用均符合本合同。
- 重放与隔离：相同 seed/版本/动作得到相同世界行为；插入读取、失败请求及重试不改变轨迹；隐藏状态配对不通过观察、动作列表或元数据提前泄漏真值。

Schema 只验证单个 JSON 值的结构及少量同帧条件，不能验证 `from_step=t-1`、跨调用状态变化、ID 唯一性、深拷贝、幂等、RNG 重放或公开信息边界。这些是实现及运行时测试的责任。JSON Schema 的 `integer` 表示数学整数，可能接受数值为整数的 `1.0`；Python 请求仍须使用真正的 `int`，不能使用 `float` 或 `bool`。非有限数字与非 JSON Python 对象也须在运行时或严格 JSON 序列化时拒绝。

Python 标准库不提供 Draft 2020-12 验证器；本 JSON Schema 是跨项目交换合同，运行时实现仍须自行执行相应校验。
