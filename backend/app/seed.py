"""All policies below are fictional. Not real company guidance."""
import json
import secrets
from pathlib import Path

from sqlalchemy import select

from .auth import hash_password
from .config import settings
from .db import Session, User, Document
from .knowledge import add_chunks


DOCUMENTS = [
('BIZ-001', '公司与代运营服务介绍', '公司业务', '## 我们是谁\n星澜电商是一家虚构的品牌电商代运营企业，为品牌客户提供店铺运营、商品管理、营销活动、客服协作和经营复盘。服务对象是已签约的品牌客户。\n## 协作链路\n销售完成签约后，实施项目启动与交接，再由运营负责日常经营。项目启动详见 FLOW-001，岗位责任详见 ORG-001。'),
('BIZ-002', '服务范围与承诺边界', '公司业务', '## 服务范围\n标准服务包含商品维护、店铺活动执行、周报与复盘、客户服务协作。广告代投只有在合同明确包含时才提供。\n## 承诺边界\n不承诺保底销售额，不替客户承担商品质量责任。新增需求、交付时间与费用必须由项目负责人确认，员工不得口头承诺。审批职责见 AUTH-001。'),
('ORG-001', '部门职责与协作关系', '岗位职责', '## 部门职责\n销售部负责商务合同与范围确认；运营部负责店铺日常运营、活动方案与上线；客服部负责消费者咨询、售后和异常记录；投放组负责广告计划与执行；项目负责人协调客户交付，运营主管审核业务方案。\n## 联系岗位\n活动审批找运营主管，合同范围找销售负责人，广告协作找投放负责人。职责索引见 CONTACT-001。'),
('JOB-001', '店铺运营专员岗位说明', '岗位职责', '## 日常工作\n运营专员维护商品、编写活动方案、检查库存与页面、提交周报，跟进活动效果。运营对店铺活动执行负责，不代替投放组调整广告预算。\n## 权限边界\n试用期授权见 HR-002；折扣与预算审批见 AUTH-001。未获审批不得对客户确认促销折扣、交付变更或新增预算。'),
('JOB-002', '客服岗位说明', '岗位职责', '## 客服职责\n客服解答消费者咨询，按已批准的活动规则提供说明，记录售后与投诉。客服不得自行修改促销规则或承诺未批准的折扣。\n## 临时变更\n客户要求变更活动时，客服记录诉求并交运营专员，按 FLOW-006 提交运营主管审核。转正不意味着获得运营审批权限。'),
('FLOW-001', '客户项目启动流程', '业务流程', '## 启动顺序\n销售确认合同范围与联系人；项目负责人组织启动会；运营接收交接清单并制定首月计划；客服接收商品与活动资料；客户确认计划后开始执行。\n## 前置资料\n需要合同服务范围、店铺授权、商品资料、库存与客户联系人。交接清单见 FLOW-002，承诺边界见 BIZ-002。'),
('FLOW-002', '销售到运营的项目交接规范', '业务流程', '## 交接清单\n销售移交合同、服务范围、客户联系人、商品清单、店铺授权状态和已确认的时间节点。运营核对资料并登记缺项。\n## 缺项处理\n缺少合同范围或授权时不得直接上线，由项目负责人协调销售补齐。不能以客户催促代替正式交接。'),
('FLOW-003', '日常运营与周报流程', '业务流程', '## 每日检查\n检查商品状态、库存、活动页面、咨询与异常。出现活动价格错误时先暂停错误入口，再按 FLOW-007 升级。\n## 周报\n每周五整理成交、流量、转化和退款指标，解释变化、记录行动与负责人。指标口径见 DATA-001，报表位置见 DATA-002。'),
('FLOW-004', '促销活动策划与上线流程', '业务流程', '## 活动准备\n运营收集客户目标，核对商品、库存、价格、优惠叠加、活动时间与素材，形成活动方案；运营主管审核，客户确认后安排上线检查。\n## 上线与协作\n未批准不得发布折扣。涉及广告则按 FLOW-005 协作；试用期执行限制见 HR-002；临时变更按 FLOW-006。'),
('FLOW-005', '广告投放协作流程', '业务流程', '## 发起投放\n运营提供活动目标、商品链接、时间、素材与拟定预算，先确认合同包含广告服务。投放组制定投放计划，运营主管与客户确认预算后由投放组执行。\n## 预算边界\n运营专员和客服不得直接修改广告预算。合同范围见 BIZ-002，预算授权见 AUTH-001。'),
('FLOW-006', '活动临时变更与审批流程', '业务流程', '## 变更处理\n收到临时改折扣、活动时间或规则的需求，先记录原因和影响范围；运营专员评估库存、价格和素材影响；运营主管审核；客户确认后再执行，并同步客服与投放组。\n## 审批角色\n客服只能记录并转交，不能直接确认变更；试用期运营需导师复核提交材料。具体权限见 AUTH-001，试用期限制见 HR-002。'),
('FLOW-007', '客诉与异常升级流程', '业务流程', '## 异常登记\n客服记录客户、订单或活动标识、问题时间、事实与截图，交运营排查；商品质量由品牌客户确认，不能承诺公司承担未约定赔偿。\n## 升级路径\n价格错误、活动配置错误立即通知运营主管；涉及交付范围和客户争议通知项目负责人。赔偿额度没有统一规定，应查看合同并交项目负责人确认。'),
('DATA-001', '常见经营指标解释', '经营知识', '## 指标口径\nGMV 为统计周期内成交总额；转化率为同口径成交人数除以访客人数；客单价为成交金额除以成交买家数。退款率必须注明分子分母与时间口径。\n## 解读边界\n不能只凭销售额下降判断广告无效，应检查流量、转化、客单价、库存和活动差异。本助手解释分析流程，不连接真实经营数据。'),
('DATA-002', '报表位置与问题分析指引', '经营知识', '## 报表入口\n公司模拟知识库约定：周报在项目工作区的经营周报目录，经营原始数据由各店铺后台导出。实际系统没有接入这些后台。\n## 分析顺序\n先统一时间和指标口径，再拆解流量、转化与客单价，检查缺货、价格、活动和投放变化，提出需验证的原因。指标定义见 DATA-001。'),
('HR-001', '新员工入职与学习指引', '入职制度', '## 第一周\n第一天了解 BIZ-001 公司业务与 ORG-001 部门职责；第二天阅读本岗位说明；第三天学习 FLOW-001 项目启动；第四天跟随导师学习活动流程；第五天向主管汇报学习问题。\n## 试用期\n导师负责任务指导和材料复核，主管负责权限和考核。岗位授权不能仅按入职时间推断，应以账号状态与 HR-002 规定为准。'),
('HR-002', '试用期工作与授权规定', '入职制度', '## 适用对象\n适用于试用期运营专员。可独立整理周报草稿、检查商品和准备活动材料。\n## 特殊限制\n对外提交活动方案前需导师复核；不能独立确认折扣、变更交付时间、修改预算或正式发布活动。导师复核不能替代主管审批。审批流程见 FLOW-006，岗位权限见 AUTH-001。'),
('AUTH-001', '运营事项审批权限规定', '入职制度', '## 折扣与活动\n促销折扣、活动规则和上线时间由运营主管审核，并获得客户确认。正式运营可提交审批，不得把提交权理解为批准权；客服没有活动审批权。\n## 试用期与预算\n试用期运营的审批材料必须先由导师复核。广告预算由投放组拟定，主管审核、客户确认后执行。公司没有通用的员工自主折扣额度。关联 FLOW-006 与 FLOW-005。'),
('CONTACT-001', '业务事项咨询岗位索引', '岗位职责', '## 咨询岗位\n活动方案与折扣：运营主管；试用期任务：导师；广告计划：投放负责人；合同范围：销售负责人；跨部门交付争议：项目负责人；账号和入职资料：人事专员。\n## 使用说明\n这里只提供负责岗位，不提供真实个人姓名或电话。角色职责见 ORG-001。'),
]


def seed():
    s = settings()
    with Session.begin() as db:
        if not db.scalar(select(User.id).limit(1)):
            if s.app_mode == 'live' and (not s.admin_password or not s.employee_password):
                raise ValueError('首次 live 启动请配置 ADMIN_PASSWORD 与 EMPLOYEE_PASSWORD')
            admin_pw = s.admin_password or 'Admin@2026'
            employee_pw = s.employee_password or 'Welcome@2026'
            for username, name, role, dept, position, probation, pw in [
                ('admin', '陈知 · 管理员', 'admin', '知识管理', '知识库管理员', False, admin_pw),
                ('operator', '林晓', 'employee', '运营部', '店铺运营专员', True, employee_pw),
                ('support', '周宁', 'employee', '客服部', '客服专员', False, employee_pw),
            ]:
                db.add(User(username=username, name=name, role=role, department=dept, position=position, probation=probation, password_hash=hash_password(pw)))
        if db.scalar(select(Document.id).limit(1)):
            return
        out = Path(s.data_dir) / 'seed_documents'
        out.mkdir(parents=True, exist_ok=True)
        for code, title, category, content in DOCUMENTS:
            content = f'# {title}\n\n> 星澜电商模拟资料，仅用于项目演示。\n\n{content}'
            special = code in ('HR-002', 'AUTH-001')
            d = Document(code=code, title=title, version='2.0' if special else '1.0', category=category,
                         effective_from='2026-07-01' if special else '2026-01-01', status='published' if s.app_mode == 'demo' else 'draft', content=content)
            db.add(d)
            db.flush()
            add_chunks(db, d)
            (out / f'{code}_v{d.version}.md').write_text(content, encoding='utf-8')
            if special:
                old_text = ('# 试用期工作与授权规定\n\n> 模拟历史制度。\n\n## 历史权限\n试用期运营提交活动材料后由主管审核，旧版未要求导师预审；仍不得独立确认折扣或上线。审批见 AUTH-001。' if code == 'HR-002' else '# 运营事项审批权限规定\n\n> 模拟历史制度。\n\n## 历史审批\n活动由运营主管审核并获得客户确认；试用期材料可直接交主管。自 2026-07-01 起按新版增加导师复核。客服无活动审批权。')
                old = Document(code=code, title=title, version='1.0', category=category, effective_from='2026-01-01', effective_to='2026-07-01', status=d.status, content=old_text)
                db.add(old)
                db.flush()
                add_chunks(db, old)
                (out / f'{code}_v1.0.md').write_text(old_text, encoding='utf-8')
