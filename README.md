# 动作捕捉处理平台

平台把三类数据从第一版开始分开保存：

- `captures/`：采集记录与原始人工标记观测，写入后不可变。
- `calibrations/`：骨架骨段长度、关节角度范围和标记绑定。
- `processing/`：每次恢复/修订产生的独立处理版本，保留输入、参数和血缘。

当前版本支持“带缺测的人工标记点恢复”和多人靠近交错时的身份修订。恢复会同时使用时间连续性、骨段长度和校准得到的关节范围；超过可接受缺口的帧保持 `null` 与 `uncertain`，不会编造完整动作。人工确认或否定按“轨迹 × 帧”保存，只清除实际决定的具体片段；无关身体部位、其他时间段和相邻确认区间继续保留 `ambiguous`。冲突决定逐项标记为 `conflicted`，`decision_effects` 列出每条决定影响的轨迹、解决或撤回后重新出现的歧义单元以及冲突帧段。撤销确认只重算受影响窗口，原始采集和旧处理版本均不可覆盖，重开后可从最新版本继续追加核对版本。动作分类和导出适配在目录结构与处理流水线中预留。

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
6. 额外生成双人交错动作：v001 保留交叉和重现歧义，v002 应用精确到轨迹和帧段的人工确认并传播到肩/肘/腕，v003 撤销确认后只让对应歧义重新出现。

主要结果在：

- `data/processing/demo_action/identity_swap_demo/versions/v001/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/versions/v002/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/evaluation_report.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/versions/v00{1,2,3}/reconstruction.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/evaluation_report.json`

双人报告会并排比较三个版本的骨长 RMSE、跨人员接触帧、身份连续性误差以及保留的歧义标记帧数量。
