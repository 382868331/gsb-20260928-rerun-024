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

### 分批补交观测

缺失区间之后可以为同一物体分批补交观测，持久化在 `processing/<action>/<pipeline>/supplements/` 下，写入后不可变：

- `SupplementBatch` 是一次补交，按物体分组，包含若干条 `SupplementTrack`（物体轨迹或带人物标记的手/身体轨迹）。轨迹按帧对齐，缺失帧保持 `null`，绝不插值。
- 补交只填补当时分析里**已经缺失**的帧：旧分析可见的原始观测永远优先，已采用补交落在原始可见帧上时只计为重复（`duplicate_frames`），不增加证据，也不会改写原始采集、物体轨迹或已确认的人工接触决定。
- 审阅者通过不可变的 `SupplementAdoption` 选择哪些缺口区间采用新资料，选择可以只覆盖补交轨迹的一部分（局部采用）；未采用的批次保存为重开后仍可见的未决证据（`pending_intervals`），并由 `load_supplement_batches`、`load_supplement_adoptions` 继续核对。
- 两人同时接触同一物体时，可为物体与两只手分别提交长短不同、部分重叠的轨迹。评估并列给出三类范围：新资料支持接触的 `supported_intervals`、多批已采用资料互相矛盾的 `conflicting_intervals`（逐帧并列双方批次、距离与理由，不静默二选一）以及补测后 `still_missing_intervals`；几何不相容仍归为几何冲突并保留补测来源。重复采用同一点位只折叠为一个证据点，被折叠的批次记录在案。
- 撤回一批补交（`revoke_supplement_batch`）只移除该批资料，依赖它（`depends_on` 含批次 id）的候选与方案进入重检（`contact_dependent_candidates`、`contact_dependent_schemes`）；从其他批次采用的区间与人工接触说明保持原样。
- 导出的 `review_export.json` 在每条接触的 `frame_observation_provenance` 中记录每一帧手与物体位置来自原始观测还是哪个补测批次，并包含全部批次、采用选择与 `supplement_review`（支持/矛盾/缺失/未决/重复范围）。

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
9. 生成接触证据复核场景：v001 保存移动球、短时物体缺口、交叉接触、共同抓握、接触后分离、短时遮挡和自动邻近推荐；v002 独立比较换位/保持标签两种方案及一个重复认领方案，采纳相容候选并排除几何冲突候选；v003 撤回一条人工接触后只重检依赖它的候选和方案。v004 接收两批长短不同、部分重叠的补测（物体 + 两只手），全部保持未决，旧缺口与已确认人工接触不变；重开后 v005 局部采用两批：重叠区间重复采用不增加证据、第 27 帧两批矛盾点位并列展示；v006 撤回较短的一批，只重检依赖它的补测候选与方案，最终导出包含原观测、批次、采用选择、候选状态、理由与逐帧观测来源的 `review_export.json`。

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
- `data/processing/demo_multi_person_action/contact_evidence_review/versions/v00{4,5,6}/reconstruction.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/{contacts,objects,candidates,schemes}/`
- `data/processing/demo_multi_person_action/contact_evidence_review/supplements/`
- `data/processing/demo_multi_person_action/contact_evidence_review/review_export.json`
- `data/processing/demo_multi_person_action/contact_evidence_review/evaluation_report.json`

双人报告会并排比较三个版本的骨长 RMSE、跨人员接触帧、身份连续性误差以及保留的歧义标记帧数量。
