# 动作捕捉处理平台

平台把三类数据从第一版开始分开保存：

- `captures/`：采集记录与原始人工标记观测，写入后不可变。
- `calibrations/`：骨架骨段长度、关节角度范围和标记绑定。
- `processing/`：每次恢复/修订产生的独立处理版本，保留输入、参数和血缘。

当前版本支持“带缺测的人工标记点恢复”和多人靠近交错时的身份修订。恢复会同时使用时间连续性、骨段长度和校准得到的关节范围；超过可接受缺口的帧保持 `null` 与 `uncertain`，不会编造完整动作。人工确认或否定会作为不可变采集之外的处理决策保存，并传播到相关关节；撤销确认只重算受影响窗口，旧处理版本继续保留。两种归属同样合理的窗口保持 `ambiguous`。动作分类和导出适配在目录结构与处理流水线中预留。

人工决定严格按对象和时间范围生效：确认右踝一小段时间只清除该决定覆盖的标记与帧，其他标记（如左腕）和其他时间段的歧义继续保持 `ambiguous`。多条决定冲突时逐条处理——先到的决定生效，冲突决定不应用并逐条记录冲突窗口、涉及标记与双方轨迹；每条决定的影响轨迹（`decision_impacts`）、各自帧范围（`decision_windows`）和撤回后重新出现的歧义（`restored_ambiguities`）都写入对应处理版本的 `parameters`。局部遮挡、人员交叉与相邻确认区间共用窗口时范围仍逐条区分。决定本身持久化在 `processing/<action>/<pipeline>/decisions/` 下，写入后不可覆盖，撤销以独立的撤销记录追加，重开后可通过 `load_decisions` 继续核对。

身份整段换位后的人工修订以“轨迹改归”（`TrajectoryReassignment`）表达：把选定时间范围内一段轨迹从一个人改归另一个人，源关节与目标关节可以不同（如 `p1` 的右臂轨迹改归为 `p0` 的左臂）。改归先以 `proposed` 状态记录并只在 `parameters.reassignment_impacts[].preview` 中展示关联变化——受影响的已有确认、相交的歧义窗口与涉及的骨架标记——不改变任何帧；审阅者决定哪些确认仍适用（`kept_decision_ids`）后才以独立的应用记录转为 `applied`。同一段轨迹被另一条改归重复采用时不会静默共用：后到者标记为 `disputed`，双方的轨迹、帧范围与共享标记逐条写入 `reassignment_disputes`，争议保留或同时调整两边都由审阅者决定。决定与改归之间的依赖显式记录（`depends_on` 与 `reassignment_dependencies`）；撤回一次改归后，依赖它的确认重新进入待处理（`pending_confirmations`），其他时间和其他人员的确认保持原样。每个版本在 `parameters.pose_comparison` 中并排比较改归窗口前后的骨长偏差与未解决区间。改归持久化在 `processing/<action>/<pipeline>/reassignments/` 下，写入后不可覆盖，应用与撤销以独立记录追加，重开后可通过 `load_reassignments` 继续审阅。最终导出的 `review_export.json` 保留原采集观测、全部改归范围与逐项决定来源。

多人交叉并同时接触同一物体时，仅靠轨迹靠近不足以判断遮挡后的身份，审阅者可以补充“接触证据”（`ContactEvidence`）：标记某人的手或脚在一段时间内接触哪个物体，证据带来源（`source`，默认 `manual_review`）与容差。接触对象本身的轨迹随采集保存（`objects` 与 `object_frames`），对象可以移动，也可以有缺失片段。比较身份解释与接触证据时逐帧区分三种情况：几何上不相容（`incompatible`，写入 `conflict_intervals` 标明冲突落在哪一段时间）、缺少观察（标记未真正观测到或对象缺失的帧列入 `missing_frames`，全部缺失则为 `unobserved`）、仍可成立（`compatible`）；`used_observations` 逐帧列出实际采用的观测。只有状态为 `observed` 的观测才作为证据，插值或不确定的位置不会让一个解释看似确定。两个人同时接触同一物体可以成立（`simultaneous_contacts`，`allowed: true`），但这绝不允许同一身体轨迹被两个身份同时认领：同一解释中重复认领同一轨迹的改归或决定会作为 `trajectory_claim_conflicts` 逐条列出。证据不足时多个解释并存，自动比较结果（`automatic_assessment`）只是推荐，不等于人工确认。

同一交叉过程的不同解释可以作为独立方案（`InterpretationScenario`）保存，各自记录采用的决定、改归与依赖的接触证据，以及已排除的候选及理由（`excluded_candidates`）。每个方案的比较结果（`scenario_assessments`）展示该方案下仍有争议的轨迹认领与接触区间。改变一个接触依据或撤回身份决定时，只重新检查依赖它的解释和确认：依赖被撤接触的决定重新进入待处理（`pending_confirmations`，原因 `contact_revoked`），`affected_scenarios` 只列出依赖被改证据的方案，其他方案不重算。接触证据持久化在 `processing/<action>/<pipeline>/contacts/` 下，方案在 `scenarios/` 下，写入后不可覆盖，撤销、采用与排除都以独立记录追加，重开后可通过 `load_contacts` 与 `load_scenarios` 继续解决争议。导出的 `review_export.json` 同时保留原观测（含对象轨迹）、人工接触说明、各方案采用或排除的身份候选及理由。

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
8. 生成轨迹改归复核场景：v001 基线；v002 确认左臂 20–24 帧；v003 提出把 `p1` 右臂轨迹 20–29 帧改归为 `p0` 左臂，仅展示关联变化预览；v004 应用改归并保留既有确认；v005 追加依赖该改归的右踝确认；v006 另一人重复采用同一腕部轨迹，标记为争议且不应用；v007 从磁盘重载后撤销两条改归，依赖它的踝部确认重新进入待处理，左臂确认与其他时段保持原样，并导出 `review_export.json`。
9. 生成接触证据复核场景（手臂交叉 + 两人先后接触同一移动物体 + 物体轨迹缺失片段 + 短时遮挡）：v001 基线，接触比较给出不相容区间与缺失帧；v002 应用接触一致的改归后接触成立；v003 保存三个解释方案——接触一致的改归方案（compatible）、不改归的朴素方案（incompatible，被排除并记录理由）、重复认领同一腕部轨迹的方案（contested，争议逐条列出）；v004 撤回一条共享接触依据，依赖它的踝部确认重新进入待处理，只重新检查依赖该接触的方案，其余方案保持不变；最后人工采用方案 A 并导出 `review_export.json`。

主要结果在：

- `data/processing/demo_action/identity_swap_demo/versions/v001/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/versions/v002/reconstruction.json`
- `data/processing/demo_action/identity_swap_demo/evaluation_report.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/versions/v00{1,2,3}/reconstruction.json`
- `data/processing/demo_multi_person_action/crossing_identity_review/evaluation_report.json`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/versions/v00{1,2,3,4,5}/reconstruction.json`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/decisions/`
- `data/processing/demo_multi_person_action/scoped_confirmation_review/evaluation_report.json`
- `data/processing/demo_multi_person_action/trajectory_reassignment_review/versions/v00{1,2,3,4,5,6,7}/reconstruction.json`
- `data/processing/demo_multi_person_action/trajectory_reassignment_review/reassignments/`
- `data/processing/demo_multi_person_action/trajectory_reassignment_review/review_export.json`
- `data/processing/demo_multi_person_action/trajectory_reassignment_review/evaluation_report.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/versions/v00{1,2,3,4}/reconstruction.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/contacts/`
- `data/processing/demo_multi_person_action/contact_evidence_review/scenarios/`
- `data/processing/demo_multi_person_action/contact_evidence_review/review_export.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/evaluation_report.json`

双人报告会并排比较三个版本的骨长 RMSE、跨人员接触帧、身份连续性误差以及保留的歧义标记帧数量。
