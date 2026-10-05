# 动作捕捉处理平台

平台把三类数据从第一版开始分开保存：

- `captures/`：采集记录与原始人工标记观测，写入后不可变。
- `calibrations/`：骨架骨段长度、关节角度范围和标记绑定。
- `processing/`：每次恢复/修订产生的独立处理版本，保留输入、参数和血缘。

当前版本支持“带缺测的人工标记点恢复”和多人靠近交错时的身份修订。恢复会同时使用时间连续性、骨段长度和校准得到的关节范围；超过可接受缺口的帧保持 `null` 与 `uncertain`，不会编造完整动作。人工确认或否定会作为不可变采集之外的处理决策保存，并传播到相关关节；撤销确认只重算受影响窗口，旧处理版本继续保留。两种归属同样合理的窗口保持 `ambiguous`。动作分类和导出适配在目录结构与处理流水线中预留。

人工决定严格按对象和时间范围生效：确认右踝一小段时间只清除该决定覆盖的标记与帧，其他标记（如左腕）和其他时间段的歧义继续保持 `ambiguous`。多条决定冲突时逐条处理——先到的决定生效，冲突决定不应用并逐条记录冲突窗口、涉及标记与双方轨迹；每条决定的影响轨迹（`decision_impacts`）、各自帧范围（`decision_windows`）和撤回后重新出现的歧义（`restored_ambiguities`）都写入对应处理版本的 `parameters`。局部遮挡、人员交叉与相邻确认区间共用窗口时范围仍逐条区分。决定本身持久化在 `processing/<action>/<pipeline>/decisions/` 下，写入后不可覆盖，撤销以独立的撤销记录追加，重开后可通过 `load_decisions` 继续核对。

身份整段换位后的人工修订以“轨迹改归”（`TrajectoryReassignment`）表达：把选定时间范围内一段轨迹从一个人改归另一个人，源关节与目标关节可以不同（如 `p1` 的右臂轨迹改归为 `p0` 的左臂）。改归先以 `proposed` 状态记录并只在 `parameters.reassignment_impacts[].preview` 中展示关联变化——受影响的已有确认、相交的歧义窗口与涉及的骨架标记——不改变任何帧；审阅者决定哪些确认仍适用（`kept_decision_ids`）后才以独立的应用记录转为 `applied`。同一段轨迹被另一条改归重复采用时不会静默共用：后到者标记为 `disputed`，双方的轨迹、帧范围与共享标记逐条写入 `reassignment_disputes`，争议保留或同时调整两边都由审阅者决定。决定与改归之间的依赖显式记录（`depends_on` 与 `reassignment_dependencies`）；撤回一次改归后，依赖它的确认重新进入待处理（`pending_confirmations`），其他时间和其他人员的确认保持原样。每个版本在 `parameters.pose_comparison` 中并排比较改归窗口前后的骨长偏差与未解决区间。改归持久化在 `processing/<action>/<pipeline>/reassignments/` 下，写入后不可覆盖，应用与撤销以独立记录追加，重开后可通过 `load_reassignments` 继续审阅。最终导出的 `review_export.json` 保留原采集观测、全部改归范围与逐项决定来源。

## 接触证据复核

多人交叉并同时接触同一物体时，轨迹靠近不足以解决遮挡后的身份。接触证据单独保存在 `processing/<action>/<pipeline>/contacts/`、`objects/`、`candidates/` 与 `schemes/` 下：

- `ContactObservation` 记录“谁的手/脚、在哪一段时间、接触哪个物体”，并带来源（如人工片段、`automatic_recommendation`）。自动推荐只作为证据，不会覆盖人工说明，也不构成人工确认。
- `ObjectTrack` 逐帧保存物体位置；物体移动或短缺口时，缺失帧保持 `null`。几何比较只使用原始观测，不使用插值补造位置。
- 同一交叉过程可保存多个独立 `IdentityScheme`。每个候选（`IdentityCandidate`）用关节到实际观测标记的映射表达一种身份解释，可标记 `adopted`、`excluded` 或继续 `open`。
- 接触比较逐帧给出三类结果：`unobserved`（手或物体缺观察）、`geometrically_incompatible`（距离超阈值）与 `still_plausible`（至少有观测相容）。结果同时列出冲突帧、缺少观察帧和实际采用的观测距离。
- 两个人同时接触同一物体作为 `simultaneous_contacts` 保留；但同一条身体轨迹在同一帧被两个身份认领时记录 `claim_conflicts`，不会因共同接触物体而允许重复认领。
- 撤回接触或改归时，只有 `depends_on` 这些依据的候选和包含它们的方案进入重检（`contact_dependent_candidates`、`contact_dependent_schemes`）。每个方案分别保留仍有争议的姿态区间、接触冲突区间和缺观察区间。
- 重开后通过 `load_contacts`、`load_object_tracks`、`load_candidates` 与 `load_schemes` 继续审阅。导出的 `review_export.json` 同时包含原观测、物体轨迹、人工接触说明、每个候选采用/排除状态、理由与逐帧相容性。

## 接触缺测补测批

已有接触缺失范围可以补交分批观测（`SupplementaryBatch`，保存在 `processing/<action>/<pipeline>/supplements/`）：同一物体可补交多批，每批为物体和两只手分别携带长短不同、部分重叠的轨迹。补测批写入后不可变，初始为 `pending`；用户用 `adopt_supplement` 按缺失区间逐段采用（可多次追加，状态变为 `partially_adopted`），用 `withdraw_supplement` 追加独立撤回记录。

- 补测只作为复核证据：旧处理版本保留当时可见的观测与缺口，补测结果不回写 `contact_review` 的判定；已采纳候选对应的人工接触决定标记 `protected_by_confirmed_decision`，即使补测矛盾也不改写。
- 逐帧事实合并：已采纳补测可以只补物体或只补一只手，并与该帧有效的原始观测或另一批已采纳补测组合；只有仍缺任一必需位置的帧保留在缺失范围。未采纳补测只列为待解释，撤回补测不参与事实合并。
- 冲突补测并列展示：`supplement_review` 逐批给出支持（`supported_frames`）、矛盾（`contradicting_frames`）、未采用（`pending_frames`）与仍缺失（`still_missing_frames`）范围；两批对同一帧结论相反时写入 `conflicting_supplements`，同一必需位置的来源不一致时写入 `source_conflicts`，均并列对照且不按到达顺序覆盖。
- 重复采用不增加证据：同一帧被多批采用时记入 `duplicate_adoptions`，证据只计一次。
- 撤回一批补测只影响依赖它的解释：`supplement_dependent_candidates` 与 `supplement_dependent_schemes` 逐条列出需重检的候选与方案，其他批次已采用的区间保持不变。
- 未决补测保存后可重开（`load_supplements`）再局部采用；导出的 `review_export.json` 中 `supplementary_batches[].frame_evidence` 逐帧记录使用了哪一批的哪条轨迹、距离与人工采用区间，可完整追溯。

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
9. 生成接触证据复核场景：v001 保存移动球、短时物体缺口、交叉接触、共同抓握、接触后分离、短时遮挡和自动邻近推荐；v002 独立比较换位/保持标签两种方案及一个重复认领方案，采纳相容候选并排除几何冲突候选；v003 撤回一条人工接触后只重检依赖它的候选和方案，并导出包含原观测、人工接触说明、候选状态与理由的 `review_export.json`。
10. 生成接触缺测补测场景：v004 保存两批未决补测（物体与两只手长短不同、部分重叠的轨迹，第二批与第一批矛盾）；v005 重开后两批各局部采用 22–23 帧，并列展示支持/矛盾/仍缺失（27 帧）范围、重复采用去重与冲突对照，并导出含逐帧补测证据的 `review_export.json`；v006 撤回第二批补测，仅依赖它的候选与方案进入重检。

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
- `data/processing/demo_multi_person_action/contact_evidence_review/versions/v00{1,2,3}/reconstruction.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/{contacts,objects,candidates,schemes,supplements}/`
- `data/processing/demo_multi_person_action/contact_evidence_review/versions/v00{4,5,6}/reconstruction.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/review_export.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/evaluation_report.json`

双人报告会并排比较三个版本的骨长 RMSE、跨人员接触帧、身份连续性误差以及保留的歧义标记帧数量。
