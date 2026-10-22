# LangGraph 多跳与有限查询改写演示

2026-10-22，通过运行中的 live API 实际执行。结果与节点事件：`data/langgraph-route-demo.json`。本次创建了两条管理员演示会话，可在“问答与反馈”查看相同问题的运行详情。

## 案例一：沿文档引用补查第二跳

复制到知序：

> 我是试用期运营。请结合试用期授权、活动审批和临时变更流程说明：客户临时改折扣时，我能否直接确认？导师复核后还需谁审批和确认，执行后同步哪些岗位？

本次实跑：

1. understand 判定 multi，拆成授权范围、审批确认、执行同步三个子问题。
2. retrieve 首轮检索包含 HR-002、JOB-001、FLOW-006 等来源。
3. hop 识别尚未覆盖的引用 AUTH-001、BIZ-002、CONTACT-001、FLOW-002、FLOW-005，并定向补查。实际返回 CONTACT-001、FLOW-002、FLOW-005、BIZ-002；AUTH-001 本次未返回，不应把引用识别等同于成功召回。
4. assess 四项检查均通过，未进入查询改写。
5. generate → verify，回答完成：不能独立确认折扣；导师复核不替代主管审批；主管审核、客户确认后执行，并同步客服及投放组。

节点路线：`understand → retrieve → hop → assess → generate → verify`。第二跳一次，查询改写零轮。

## 案例二：证据不足，改写两轮后停止

复制到知序：

> 2025年12月31日，试用期运营能否独立确认促销折扣？请查找当时有效的权限制度和活动审批流程，给出具体条款。

本次知识库没有覆盖该日期的有效制度，因此能观察到证据不足分支：

1. 首轮检索无来源，assess 标记缺失岗位权限和活动审批流程。
2. 第一次 rewrite：
   “截至2025年12月31日，试用期内的运营岗位人员是否被授权单独批准促销折扣？请引用当时有效的岗位权限管理办法及促销活动审批流程中规定折扣审批权限层级、审批人资格和试用期人员限制的条款。”
3. supplement 返回零条，assess 仍不满足。
4. 第二次 rewrite：
   “2025年12月31日施行的运营岗位权限管理制度中，试用期员工在促销折扣审批上是否被列为无独立确认权的角色，对应条款原文是什么？”
5. supplement 仍返回零条，assess 记录 retry_count=2、stop_reason=达到查询改写重试上限，不再改写。
6. 生成无法可靠确认该日期规则的回答。没有把 2026 年制度套用于 2025 年。

节点路线：`understand → retrieve → hop(无依赖不补查) → assess → rewrite → supplement → assess → rewrite → supplement → assess → generate → verify`。

本次首次运行在 generate 出现模型 JSON 解析失败；随后通过检查点恢复成功，没有重复两轮改写/补查。恢复前记录为 `data/langgraph-route-demo-before-resume.json`。

## 如何查看、复现

- 页面：管理员登录 → 问答与反馈 → 查找以上问题 → 运行详情，查看 hop/dependency_refs、rewrite/query、supplement/retry_number 和 assess/stop_reason。
- 重新创建演示会话：`python -m scripts.demo_langgraph_routes`。
- 从上次保存的运行恢复：`python -m scripts.demo_langgraph_routes --resume`。
- 实跑是动态模型判断，语料或模型输出改变后，第一轮就满足时会提前结束，不保证任何问题都强制走二跳或两轮重试。
- “最多两轮”指首轮之后的查询改写补查轮数；首轮子问题检索、第二跳另行计数，并同时受 MAX_RETRIEVAL_CALLS=6 限制。

## 代码文件夹

| 文件夹 | 文件 | 作用 |
|---|---|---|
| backend/app | workflow.py | LangGraph 节点、第二跳、证据检查、改写、重试分支 |
| backend/app | retrieval.py | BM25L、Milvus HNSW、RRF、本地 BGE 重排及原文补充 |
| backend/app | config.py | MAX_QUERY_RETRIES 默认 2、MAX_RETRIEVAL_CALLS 默认 6 |
| backend/tests | test_workflow.py | 可稳定复现的二跳、改写恢复证据、上限及重复停止测试 |
| scripts | demo_langgraph_routes.py | 调用实际 API 的两条演示及检查点恢复 |
| data | langgraph-route-demo.json | 本次两条实际问题、回答、节点事件与轮次 |

分支自动化验证：`python -m pytest backend/tests/test_workflow.py -q`。其中模型和检索器使用替身，验证路由及限制；与上面的真实 API 实跑记录分别保留。
