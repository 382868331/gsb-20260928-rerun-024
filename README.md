# 动作捕捉处理平台

平台把三类数据从第一版开始分开保存：

- `captures/`：采集记录与原始人工标记观测，写入后不可变。
- `calibrations/`：骨架骨段长度、关节角度范围和标记绑定。
- `processing/`：每次恢复/修订产生的独立处理版本，保留输入、参数和血缘。

当前版本支持“带缺测的人工标记点恢复”和多人靠近交错时的身份修订。恢复会同时使用时间连续性、骨段长度和校准得到的关节范围；超过可接受缺口的帧保持 `null` 与 `uncertain`，不会编造完整动作。人工确认或否定会作为不可变采集之外的处理决策保存，并传播到相关关节；撤销确认只重算受影响窗口，旧处理版本继续保留。两种归属同样合理的窗口保持 `ambiguous`。动作分类和导出适配在目录结构与处理流水线中预留。

人工决定严格按对象和时间范围生效：确认右踝一小段时间只清除该决定覆盖的标记与帧，其他标记（如左腕）和其他时间段的歧义继续保持 `ambiguous`。多条决定冲突时逐条处理——先到的决定生效，冲突决定不应用并逐条记录冲突窗口、涉及标记与双方轨迹；每条决定的影响轨迹（`decision_impacts`）、各自帧范围（`decision_windows`）和撤回后重新出现的歧义（`restored_ambiguities`）都写入对应处理版本的 `parameters`。局部遮挡、人员交叉与相邻确认区间共用窗口时范围仍逐条区分。决定本身持久化在 `processing/<action>/<pipeline>/decisions/` 下，写入后不可覆盖，撤销以独立的撤销记录追加，重开后可通过 `load_decisions` 继续核对。

## 快速运行

```bash
PYTHONPATH=src python -m mocap_platform demo --output data
PYTHONPATH=src python -m unittest discover -s tests -v
```

演示会生成一段已知骨架动作，加入短暂遮挡、一个长缺口和标记交换，然后：

1. 保存采集记录与原始观测。
2. 从未损坏的已知动作生成校准 v1。
3. 基于损坏观测生成处理 v1。
4. 修订标记身份，只重算受影响时间窗。
5. 生成处理 v2，并将恢复位置与真实动作比较。
6. 额外生成双人交错动作：v001 保留交叉和重现歧义，v002 应用人工确认并传播到肩/肘/腕，v003 撤销确认后重新生成同一窗口。
7. 生成限定范围复核场景（手臂交叉 + 右踝短时交换 + 局部遮挡 + 重现歧义）：v001 基线；v002 仅确认右踝 25–29 帧，左腕与其他时段歧义保留；v003 追加相邻区间（20–24 帧）左臂确认，两个决定窗口分别记录；v004 引入冲突的否定决定，逐条记录且不应用；v005 撤销踝部确认并演示从磁盘重载决定后继续核对，被撤窗口的歧义重新出现。

主要结果在：

- `data/processing/demo_action/identity_swap_demo/versions/v001/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/versions/v002/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/evaluation_report.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/versions/v00{1,2,3}/reconstruction.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/evaluation_report.json`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/versions/v00{1,2,3,4,5}/reconstruction.json`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/decisions/`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/evaluation_report.json`

双人报告会并排比较三个版本的骨长 RMSE、跨人员接触帧、身份连续性误差以及保留的歧义标记帧数量。
