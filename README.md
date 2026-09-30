# 动作捕捉处理平台

平台把三类数据从第一版开始分开保存：

- `captures/`：采集记录与原始人工标记观测，写入后不可变。
- `calibrations/`：骨架骨段长度、关节角度范围和标记绑定。
- `processing/`：每次恢复/修订产生的独立处理版本，保留输入、参数和血缘。

当前版本在“带缺测的人工标记点恢复”之上加入**双人靠近交错时的身份修订**：

- 恢复同时使用时间连续性、骨段长度和校准得到的关节范围；超过可接受缺口的帧保持 `null` 与 `uncertain`，不会编造完整动作。
- 标记按 `pA_`/`pB_` 人名前缀区分两人，手臂/腿部分成可整体交换的肢体组（肩-肘-腕、髋-膝-踝）。
- 人工可以在一段时间窗口内**确认或否定**某组标记归属；一次决策沿肢体组传播到相关关节，只写入新的处理版本，`captures/` 下的原始观测永不改变。
- 两人同时短遮挡时先插值恢复；遮挡结束后若两种归属在骨长证据上**同样合理**（镜像再现、标记靠近且刚发生双侧遮挡），相关远端关节保持 `uncertain`，并在 `ambiguities` 中记录两套假设的得分，等待人工裁决。
- 长缺口不编造；撤销一条人工确认会只重算“仍有效决策 ∪ 被撤销窗口”的受影响时间窗，其余处理版本（v001/v002/…）原样保留可复查。

动作分类和导出适配在目录结构与处理流水线中预留。

## 快速运行

```bash
PYTHONPATH=src python -m mocap_platform demo --output data
PYTHONPATH=src python -m unittest discover -s tests -v
```

演示会生成一段**双人交错的人工动作**（手臂身份交叉 → 双侧手腕同时短遮挡 → 镜像对称再现 → 单侧长遮挡），然后：

1. 保存采集记录与原始观测。
2. 从未损坏的已知动作生成校准 v1。
3. 基于损坏观测生成处理 v1：短遮挡被恢复、长缺口与等权归属保持 `uncertain`。
4. v2 记录两条人工决策：否定交错段的错误归属（传播到整条手臂）、确认再现段归属。
5. v3 撤销再现确认，仅重新生成受影响窗口；被否定的交错段保持正确。
6. 对 v1/v2/v3 比较骨长误差、接触恢复和身份连续性。

主要结果在：

- `data/processing/demo_action/two_person_crossing/versions/v001/reconstruction.json`
- `data/processing/demo_action/two_person_crossing/versions/v002/reconstruction.json`
- `data/processing/demo_action/two_person_crossing/versions/v003/reconstruction.json`
- `data/processing/demo_action/two_person_crossing/evaluation_report.json`

每个版本的 `reconstruction.json` 包含 `frames`、`statuses`、`identity_revisions`、`ambiguities` 和 `parameters.recompute_window`；`ambiguities[].resolution` 为 `unresolved` 或 `manual:<markers>`。
